from torch import optim

from basic_utils.utils import *


# ================================================================
# Trust Region Policy Optimization Updater
# ================================================================
class TRPO_Updater:
    def __init__(self, net, probtype, max_kl, cg_damping, cg_iters, get_info):
        self.net = net
        self.probtype = probtype
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.get_info = get_info

    def conjugate_gradients(self, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Avp = self.Fvp(p)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        fval = self.get_loss().data
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.net, xnew)
            newfval = self.get_loss().data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio[0] > accept_ratio and actual_improve[0] > 0:
                return True, xnew
        return False, x

    def Fvp(self, v):
        kl = self.probtype.kl(self.fixed_dist, self.net(self.observations)).mean()
        grads = torch.autograd.grad(kl, self.net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data.cpu()

        return flat_grad_grad_kl + v * self.cg_damping

    def get_loss(self):
        prob = self.net(self.observations)
        prob = self.probtype.likelihood(self.actions, prob)
        action_loss = -self.advantages * prob / self.fixed_prob
        return action_loss.mean()

    def _derive_info(self, observations, actions, advantages, fixed_dist, fixed_prob):
        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        surr = torch.mean(prob_ratio * advantages)
        losses = {"surr": -surr.data[0], "kl": self.probtype.kl(fixed_dist, new_prob).data.mean(),
                  "ent": self.probtype.entropy(new_prob).data.mean()}

        return losses

    def __call__(self, path):
        self.observations = turn_into_cuda(path["observation"])
        self.actions = turn_into_cuda(path["action"])
        self.advantages = turn_into_cuda(path["advantage"])
        self.fixed_dist = self.net(self.observations).detach()
        self.fixed_prob = self.probtype.likelihood(self.actions, self.fixed_dist).detach()

        if self.get_info:
            info_before = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist,
                                            self.fixed_prob)

        loss = self.get_loss()
        grads = torch.autograd.grad(loss, self.net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data.cpu()

        stepdir = self.conjugate_gradients(-loss_grad, self.cg_iters)
        shs = 0.5 * (stepdir * self.Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = turn_into_cuda(stepdir / lm[0])
        neggdotstepdir = turn_into_cuda((-loss_grad * stepdir).sum(0, keepdim=True))
        prev_params = get_flat_params_from(self.net)
        success, new_params = self.linesearch(prev_params, fullstep, neggdotstepdir / lm[0])
        set_flat_params_to(self.net, new_params)

        if self.get_info:
            info_after = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist,
                                           self.fixed_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Adam Updater
# ================================================================
class Adam_Updater:
    def __init__(self, net, probtype, lr, epochs, kl_target, get_info):
        self.net = net
        self.probtype = probtype
        self.lr = lr
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)
        self.kl_target = kl_target
        self.get_info = get_info
        self.epochs = epochs

    def _derive_info(self, observations, actions, advantages, fixed_dist):
        prob = self.net(observations)
        surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
        losses = {"surr": surr.data[0], "kl": self.probtype.kl(fixed_dist, prob).mean().data[0],
                  "ent": self.probtype.entropy(prob).mean().data[0], 'lr': self.lr}

        return losses

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        old_prob = self.net(observations).detach()

        if self.get_info:
            info_before = self._derive_info(observations, actions, advantages, old_prob)

        prob = self.net(observations)
        surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
        self.net.zero_grad()
        surr.backward()
        self.optimizer.step()

        prob = self.net(observations)
        kl = self.probtype.kl(old_prob, prob).mean().data[0]
        if kl > 4 * self.kl_target:
            self.lr /= 1.5
            self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr)
        if kl < 0.25 * self.kl_target:
            self.lr *= 1.5
            self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr)

        if self.get_info:
            info_after = self._derive_info(observations, actions, advantages, old_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Ppo Updater
# ================================================================
class PPO_adapted_Updater:
    def __init__(self, net, probtype, beta, kl_cutoff_coeff, kl_target, epochs, lr, beta_range, adj_thres,
                 get_info=True):
        self.net = net
        self.probtype = probtype
        self.beta = beta  # dynamically adjusted D_KL loss multiplier
        self.eta = kl_cutoff_coeff
        self.kl_targ = kl_target
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.get_info = get_info
        self.beta_upper = beta_range[1]
        self.beta_lower = beta_range[0]
        self.beta_adj_thres = adj_thres

    def _derive_info(self, observes, actions, advantages, old_prob):
        prob = self.net(observes)
        logp = self.probtype.loglikelihood(actions, prob)
        logp_old = self.probtype.loglikelihood(actions, old_prob)
        kl = self.probtype.kl(old_prob, prob).mean()
        surr = -(advantages * (logp - logp_old).exp()).mean()
        loss = surr + self.beta * kl

        if kl.data[0] - 2.0 * self.kl_targ > 0:
            loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

        entropy = self.probtype.entropy(prob).mean()
        info = {'loss': loss.data[0], 'surr': surr.data[0], 'kl': kl.data[0], 'entropy': entropy.data[0],
                'beta_pen': self.beta, 'lr': self.lr}

        return info

    def __call__(self, path):
        observes = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])

        old_prob = self.net(observes).detach()

        if self.get_info:
            info_before = self._derive_info(observes, actions, advantages, old_prob)

        for e in range(self.epochs):
            prob = self.net(observes)
            logp = self.probtype.loglikelihood(actions, prob)
            logp_old = self.probtype.loglikelihood(actions, old_prob)
            kl = self.probtype.kl(old_prob, prob).mean()
            surr = -(advantages * (logp - logp_old).exp()).mean()
            loss = surr + self.beta * kl

            if kl.data[0] - 2.0 * self.kl_targ > 0:
                loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

            prob = self.net(observes)
            kl = self.probtype.kl(old_prob, prob).mean()
            if kl.data[0] > self.kl_targ * 4:
                break
        if kl.data[0] > self.kl_targ * self.beta_adj_thres[1]:
            if self.beta_upper > self.beta:
                self.beta = self.beta * 1.5
            if self.beta > self.beta_upper / 1.5:
                self.lr /= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres[0]:
            if self.beta_lower < self.beta:
                self.beta = self.beta / 1.5
            if self.beta < self.beta_lower * 1.5:
                self.lr *= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob)
            return merge_before_after(info_before, info_after)


class PPO_clip_Updater:
    def __init__(self, net, probtype, epsilon, kl_target, epochs, adj_thres, clip_range, lr, get_info=True):
        self.net = net
        self.probtype = probtype
        self.clip_epsilon = epsilon
        self.kl_target = kl_target
        self.epochs = epochs
        self.get_info = get_info
        self.clip_adj_thres = adj_thres
        self.clip_upper = clip_range[1]
        self.clip_lower = clip_range[0]
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def _derive_info(self, observations, actions, advantages, fixed_dist, fixed_prob):
        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr = -prob_ratio * advantages
        cliped_surr = -cliped_ratio * advantages
        clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()
        kl = self.probtype.kl(fixed_dist, new_prob).mean()

        losses = {"surr": surr.mean().data[0], "clip_surr": clip_loss.data[0], "kl": kl.data[0],
                  "ent": self.probtype.entropy(new_prob).data.mean(), 'clip_epsilon': self.clip_epsilon, 'lr': self.lr}
        return losses

    def __call__(self, path):
        observes = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        old_prob = self.net(observes).detach()
        fixed_prob = self.probtype.likelihood(actions, old_prob).detach()

        if self.get_info:
            info_before = self._derive_info(observes, actions, advantages, old_prob, fixed_prob)

        for e in range(self.epochs):
            new_prob = self.net(observes)
            new_p = self.probtype.likelihood(actions, new_prob)
            prob_ratio = new_p / fixed_prob
            cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surr = -prob_ratio * advantages
            cliped_surr = -cliped_ratio * advantages
            clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()

            self.net.zero_grad()
            clip_loss.backward()
            self.optimizer.step()

            prob = self.net(observes)
            kl = self.probtype.kl(old_prob, prob).mean()
            if kl.data[0] > 4 * self.kl_target:
                break

        if kl.data[0] > self.kl_target * self.clip_adj_thres[1]:
            if self.clip_lower < self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon / 1.2
            if self.clip_epsilon < self.clip_lower * 1.2:
                self.lr /= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif kl.data[0] < self.kl_target * self.clip_adj_thres[0]:
            if self.clip_upper > self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon * 1.2
            if self.clip_epsilon > self.clip_upper / 1.2:
                self.lr *= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob, fixed_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Evolution Updater
# ================================================================
class Evolution_Updater:
    def __init__(self, net, n_kid, lr, sigma, get_info=True):
        self.net = net
        self.n_kid = n_kid
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.get_info = get_info
        self.sigma = sigma
        base = self.n_kid * 2
        rank = np.arange(1, base + 1)
        util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
        self.utility = util_ / util_.sum() - 1 / base

    def __call__(self, path):
        noise_seed = path[0]['seed']
        path_info = [p['reward_raw'] for p in path]
        path_index = [p['index'] for p in path]
        total_reward = np.zeros((2 * self.n_kid,))
        for i in range(len(path_index)):
            total_reward[path_index[i]] += np.sum(path_info[i])

        kids_rank = np.argsort(total_reward)[::-1]

        flat_params = get_flat_params_from(self.net).cpu().numpy()
        cumulative_update = np.zeros_like(flat_params)
        for ui, k_id in enumerate(kids_rank):
            np.random.seed(noise_seed[k_id])  # reconstruct noise using seed
            cumulative_update += self.utility[ui] * sign(k_id) * np.random.randn(flat_params.size)
        cumulative_update /= -2 * self.n_kid * self.sigma
        self.net.zero_grad()
        set_flat_grads_to(self.net, turn_into_cuda(torch.from_numpy(cumulative_update)))
        self.optimizer.step()

        return {}


# ================================================================
# Adam Optimizer
# ================================================================
class Adam_Optimizer:
    def __init__(self, net, lr, epochs, batch_size, get_data=True):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr)
        self.epochs = epochs
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.get_data = get_data
        self.default_batch_size = batch_size

    def _derive_info(self, observations, y_targ):
        y_pred = self.net(observations)
        explained_var = 1 - torch.var(y_targ - y_pred) / torch.var(y_targ)
        loss = (y_targ - y_pred).pow(2).mean()
        info = {'explained_var': explained_var.data[0], 'loss': loss.data[0]}
        return info

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        y_targ = turn_into_cuda(path["return"])

        if self.get_data:
            info_before = self._derive_info(observations, y_targ)

        num_batches = max(observations.size()[0] // self.default_batch_size, 1)
        batch_size = observations.size()[0] // num_batches

        if self.replay_buffer_x is None:
            x_train, y_train = observations, y_targ
        else:
            x_train = torch.cat([observations, self.replay_buffer_x], dim=0)
            y_train = torch.cat([y_targ, self.replay_buffer_y], dim=0)
        self.replay_buffer_x = observations
        self.replay_buffer_y = y_targ

        for e in range(self.epochs):
            sortinds = np.random.permutation(observations.size()[0])
            sortinds = turn_into_cuda(np_to_var(sortinds).long())
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs_ph = x_train.index_select(0, sortinds[start:end])
                val_ph = y_train.index_select(0, sortinds[start:end])
                out = self.net(obs_ph)
                loss = (out - val_ph).pow(2).mean()
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.get_data:
            info_after = self._derive_info(observations, y_targ)
            return merge_before_after(info_before, info_after)


# ================================================================
# Adam Q-Learning Optimizer
# ================================================================
class Adam_Q_Optimizer:
    def __init__(self, net, target_net, lr, gamma, update_target_every, get_data=True):
        self.net = net
        self.target_net = target_net
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.count = 0
        self.get_data = get_data
        self.update_target_every = update_target_every

    def _derive_info(self, observations, y_targ, actions):
        y_pred = self.net(observations).gather(1, actions.long())
        explained_var = 1 - torch.var(y_targ - y_pred) / torch.var(y_targ)
        loss = (y_targ - y_pred).pow(2).mean()
        info = {'explained_var': explained_var.data[0], 'loss': loss.data[0]}
        return info

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        weights = turn_into_cuda(path["weights"]) if "weights" in path else None
        y_targ = turn_into_cuda(path['y_targ'])

        if self.get_data:
            info_before = self._derive_info(observations, y_targ, actions)

        td_err = torch.abs(self.net(observations).gather(1, actions.long()) - y_targ)
        loss = (td_err.pow(2) * weights).sum() if weights is not None else td_err.pow(2).mean()
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if self.get_data:
            info_after = self._derive_info(observations, y_targ, actions)
            return merge_before_after(info_before, info_after), {"td_err": td_err.data.cpu().numpy()}

        return None, {"td_err": td_err.data.cpu().numpy()}
