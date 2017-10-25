from torch import optim

from basic_utils.utils import *


# ================================================================
# Trust Region Policy Optimization Updater
# ================================================================
class Trpo_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.max_kl = cfg["max_kl"]
        self.cg_damping = cfg["cg_damping"]
        self.cg_iters = cfg["cg_iters"]
        self.update_threshold = cfg["update_threshold"]
        self.get_info = cfg["get_info"]

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
            info_before = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist, self.fixed_prob)

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
            info_after = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist, self.fixed_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Adam Updater
# ================================================================
class Adam_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=cfg["lr_updater"])
        self.update_threshold = cfg["update_threshold"]
        self.kl_target = cfg["kl_target"]
        self.get_info = cfg["get_info"]
        self.epochs = cfg["epochs_updater"]

    def _derive_info(self, observations, actions, advantages, fixed_dist):
        prob = self.net(observations)
        surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
        losses = {"surr": surr.data[0], "kl": self.probtype.kl(fixed_dist, prob).mean().data[0],
                  "ent": self.probtype.entropy(prob).mean().data[0]}

        return losses

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        old_prob = self.net(observations).detach()

        if self.get_info:
            info_before = self._derive_info(observations, actions, advantages, old_prob)

        for e in range(self.epochs):
            prob = self.net(observations)
            surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
            self.net.zero_grad()
            surr.backward()
            self.optimizer.step()

            kl = self.probtype.kl(old_prob, prob).mean().data[0]
            if kl > 4 * self.kl_target:
                break

        if self.get_info:
            info_after = self._derive_info(observations, actions, advantages, old_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Ppo Updater
# ================================================================
class Ppo_adapted_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.beta = cfg["beta_init"]  # dynamically adjusted D_KL loss multiplier
        self.eta = cfg["kl_cutoff_coeff"]
        self.kl_targ = cfg["kl_target"]
        self.epochs = cfg["epochs_updater"]
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg["lr_updater"])
        self.get_info = cfg["get_info"]
        self.beta_upper = cfg["beta_upper"]
        self.beta_lower = cfg["beta_lower"]
        self.beta_adj_thres_u = cfg["beta_adj_thres_u"]
        self.beta_adj_thres_l = cfg["beta_adj_thres_l"]

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
                'beta_pen': self.beta}

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
        if kl.data[0] > self.kl_targ * self.beta_adj_thres_u and self.beta < self.beta_upper:
            self.beta = 1.5 * self.beta
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres_l and self.beta > self.beta_lower:
            self.beta = self.beta / 1.5

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob)
            return merge_before_after(info_before, info_after)


class Ppo_clip_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.clip_epsilon = cfg["clip_epsilon"]
        self.kl_target = cfg["kl_target"]
        self.update_threshold = cfg["update_threshold"]
        self.epochs = cfg["epochs_updater"]
        self.get_info = cfg["get_info"]
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg["lr_updater"])

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
                  "ent": self.probtype.entropy(new_prob).data.mean()}
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

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob, fixed_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Adam Optimizer
# ================================================================
class Adam_Optimizer:
    def __init__(self, net, cfg):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), cfg['lr_optimizer'])
        self.epochs = cfg["epoches_optimizer"]
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.get_data = cfg['get_info']
        self.default_batch_size = cfg['batch_size_optimizer']

    def _derive_info(self, observations, y_targ):
        y_pred = self.net(observations)
        explained_var = 1 - torch.var(y_targ - y_pred) / torch.var(y_targ)
        loss = (y_targ - y_pred).pow(2).mean()
        info = {'explained_variance': explained_var.data[0], 'loss': loss.data[0]}
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
            sortinds = turn_into_cuda(torch.from_numpy(sortinds).long())
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
    def __init__(self, net, target_net, cfg, double):
        self.net = net
        self.target_net = target_net
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=cfg["lr_optimizer"])
        self.gamma = cfg["gamma"]
        self.double = double
        self.count = 0
        self.update_target_every = cfg["update_target_every"]

    def __call__(self, path):
        observations = path["observation"]
        actions = path["action"]
        next_observations = path["next_observation"]
        rewards = path["reward"]
        not_dones = path["not_done"]
        out = OrderedDict()

        if not self.double:
            y_targ = self.target_net(next_observations).max(dim=1, keepdim=True)[0]
        else:
            ty = self.net(next_observations).max(dim=1, keepdim=True)[1]
            y_targ = self.target_net(next_observations).gather(1, ty.long())

        y_targ = y_targ * not_dones * self.gamma + rewards
        td_err = torch.abs(self.net(observations).gather(1, actions.long()) - y_targ)
        loss = (td_err.pow(2) * path["weights"]).sum() if "weights" in path else td_err.pow(2).mean()
        out['loss_before'] = loss.data[0]

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_after = torch.abs(self.net(observations).gather(1, actions.long()) - y_targ)
        loss = (td_after.pow(2) * path["weights"]).sum() if "weights" in path else td_after.pow(2).mean()
        out['loss_after'] = loss.data[0]

        self.count += 1
        if self.count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return out, {"td_err": td_err.data.cpu().numpy()}
