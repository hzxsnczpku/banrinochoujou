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
        self.new_params = None

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

    def step(self, flat_params=None):
        if flat_params is None:
            set_flat_params_to(self.net, self.new_params)
            self.new_params = None
        elif self.new_params is None:
            self.new_params = flat_params / self.update_threshold
        else:
            self.new_params += flat_params / self.update_threshold

    def derive_data(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])
        fixed_prob = self.probtype.likelihood(actions, fixed_dist).detach()

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
        self.fixed_dist = turn_into_cuda(path["prob"])
        self.fixed_prob = self.probtype.likelihood(self.actions, self.fixed_dist).detach()

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
        return new_params


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

    def step(self, grads=None):
        if grads is None:
            self.optimizer.step()
            self.net.zero_grad()
        else:
            for k, l in zip(self.net.parameters(), grads):
                ave_l = l / self.update_threshold
                k.grad = (k.grad + ave_l) if k.grad is not None else ave_l

    def derive_data(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])

        prob = self.net(observations).cpu()
        surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
        losses = {"surr": surr.data[0], "kl": self.probtype.kl(fixed_dist, prob).mean().data[0],
                  "ent": self.probtype.entropy(prob).mean().data[0]}

        return losses

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])

        prob = self.net(observations)
        kl = self.probtype.kl(fixed_dist, prob).mean().data[0]
        if kl > 4 * self.kl_target:
            return None

        surr = -(self.probtype.loglikelihood(actions, prob) * advantages).mean()
        self.net.zero_grad()
        surr.backward()
        grads = [k.grad for k in self.net.parameters()]
        return grads


# ================================================================
# Ppo Updater
# ================================================================
class Ppo_adapted_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.kl_beta = 1.0
        self.kl_cutoff = cfg["kl_target"] * 2.0
        self.kl_cutoff_coeff = cfg["kl_cutoff_coeff"]
        self.kl_target = cfg["kl_target"]
        self.update_threshold = cfg["update_threshold"]
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=cfg["lr_updater"])

    def step(self, grads=None, update=False):
        if update:
            self.optimizer.step()
            self.net.zero_grad()
        else:
            for k, l in zip(self.net.parameters(), grads):
                ave_l = l / self.update_threshold
                k.grad = (k.grad + ave_l) if k.grad is not None else ave_l

    def derive_data(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])
        fixed_prob = self.probtype.likelihood(actions, fixed_dist).detach()

        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        surr = -torch.mean(prob_ratio * advantages)
        kl = self.probtype.kl(fixed_dist, new_prob).mean()
        surr_pen = surr + self.kl_beta * kl
        if (kl > self.kl_cutoff).data[0]:
            surr_pen += self.kl_cutoff_coeff * (kl - self.kl_cutoff).pow(2)

        if kl.data[0] > 1.3 * self.kl_target and self.kl_beta < 35:
            self.kl_beta *= 1.5
        elif kl.data[0] < 0.7 * self.kl_target and kl.data[0] > 1e-10 and self.kl_beta > 1 / 35:
            self.kl_beta /= 1.5

        losses = {"surr": surr.data[0], "surr_pen": surr_pen.data[0], "kl": kl.data[0],
                  "ent": self.probtype.entropy(new_prob).data.mean(),
                  "kl_beta": self.kl_beta}

        return losses

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])
        fixed_prob = self.probtype.likelihood(actions, fixed_dist).detach()

        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        surr = -torch.mean(prob_ratio * advantages)
        kl = self.probtype.kl(fixed_dist, new_prob).mean()
        surr_pen = surr + self.kl_beta * kl
        if (kl > self.kl_cutoff).data[0]:
            surr_pen += self.kl_cutoff_coeff * (kl - self.kl_cutoff).pow(2)

        self.net.zero_grad()
        surr_pen.backward()
        grads = [k.grad for k in self.net.parameters()]
        return grads


class Ppo_clip_Updater:
    def __init__(self, net, probtype, cfg):
        self.net = net
        self.probtype = probtype
        self.clip_epsilon = cfg["clip_epsilon"]
        self.kl_target = cfg["kl_target"]
        self.update_threshold = cfg["update_threshold"]
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg["lr_updater"])

    def step(self, grads=None, update=False):
        if update:
            self.optimizer.step()
            self.net.zero_grad()
        else:
            for k, l in zip(self.net.parameters(), grads):
                ave_l = l / self.update_threshold
                k.grad = (k.grad + ave_l) if k.grad is not None else ave_l

    def derive_data(self, path):
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])
        fixed_prob = self.probtype.likelihood(actions, fixed_dist).detach()

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
        observations = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        fixed_dist = turn_into_cuda(path["prob"])
        fixed_prob = self.probtype.likelihood(actions, fixed_dist).detach()

        new_prob = self.net(observations)
        kl = self.probtype.kl(fixed_dist, new_prob).mean()
        if kl.data[0] > 4 * self.kl_target:
            return None
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr = -prob_ratio * advantages
        cliped_surr = -cliped_ratio * advantages
        clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()

        self.net.zero_grad()
        clip_loss.backward()
        grads = [k.grad for k in self.net.parameters()]
        return grads


# ================================================================
# Adam Optimizer
# ================================================================
class Adam_Optimizer:
    def __init__(self, net, cfg):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg["lr_optimizer"])
        self.update_threshold = cfg["update_threshold"]

    def step(self, grads=None, update=False):
        if update:
            self.optimizer.step()
            self.net.zero_grad()
        else:
            for k, l in zip(self.net.parameters(), grads):
                ave_l = l / self.update_threshold
                k.grad = (k.grad + ave_l) if k.grad is not None else ave_l

    def derive_data(self, path):
        observations = turn_into_cuda(path["observation"])
        y_targ = turn_into_cuda(path["return"])

        y_pred = self.net(observations)
        td = y_pred - y_targ
        loss = td.pow(2).mean()
        exp_var = 1 - np.var(td.data.cpu().numpy()) / np.var(y_targ.data.cpu().numpy())
        return {"loss": loss.data[0], "explainedvar": exp_var}

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        y_targ = turn_into_cuda(path["return"])

        td = self.net(observations) - y_targ
        loss = td.pow(2).mean()

        self.net.zero_grad()
        loss.backward()
        grads = [k.grad for k in self.net.parameters()]
        return grads


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
