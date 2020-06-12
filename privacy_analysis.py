# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



import os
import math
import numpy as np
import argparse

def compute_q_noisy_max(counts, noise_eps):
    """returns ~ Pr[outcome != winner].
    Args:
    counts: a list of scores
    noise_eps: privacy parameter for noisy_max (gamma)
    Returns:
    q: the probability that outcome is different from true winner.
    """
    # For noisy max, we only get an upper bound.
    # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
    # proof at http://mathoverflow.net/questions/66763/
    # tight-bounds-on-probability-of-sum-of-laplace-random-variables

    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])
    #print("counts normalized:", counts_normalized)
    counts_rest = np.array(
        [counts_normalized[i] for i in range(len(counts)) if i != winner])
    q = 0.0
    for c in counts_rest:
        gap = -c
        q += (gap + 2.0) / (4.0 * math.exp(gap))
    # print("neq q:", q)
    return min(q, 1.0 - (1.0/len(counts)))


def compute_q_noisy_max_approx(counts, noise_eps):
    """returns ~ Pr[outcome != winner].
    Args:
    counts: a list of scores
    noise_eps: privacy parameter for noisy_max (gamma)
    Returns:
    q: the probability that outcome is different from true winner.
    """
    # For noisy max, we only get an upper bound.
    # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
    # proof at http://mathoverflow.net/questions/66763/
    # tight-bounds-on-probability-of-sum-of-laplace-random-variables
    # This code uses an approximation that is faster and easier
    # to get local sensitivity bound on.

    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])
    counts_rest = np.array(
        [counts_normalized[i] for i in range(len(counts)) if i != winner])
    gap = -max(counts_rest)
    q = (len(counts) - 1) * (gap + 2.0) / (4.0 * math.exp(gap))
    return min(q, 1.0 - (1.0/len(counts)))


def logmgf_exact(q, noise_eps, l, t, z, k, max_q):
    """Computes the logmgf value given q and privacy eps.
    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf by extending to k partitioning.
    The second term is from our paper.
    The third term comes directly from the privacy guarantee.
    Args:
        q: pr of non-optimal outcome
        gamma:
        l: moment to compute.
        t: the instance portion of each party
        z: the number of partitioning that may be affected in each party
        k: number of partitioning in each party
    Returns:
        Upper bound on logmgf
    """
    priv_eps = 2 * k * noise_eps
    if q < max_q:
        part1 = (1-q) * math.pow((1-q) / (1-np.sum(t*np.exp(2*z*noise_eps)*q)), l)
        part2 = q / np.sum(t*np.exp(-2*z*noise_eps*l))
        try:
            log_t = math.log(part1 + part2)
        except ValueError:
            print("part1: ", part1)
            print("part2: ", part2)
            print("Got ValueError in math.log for values :" + str((q, noise_eps, l, t, z, k)))
            log_t = priv_eps * l
        # print("log_t:", log_t)
        # if log_t < 0.5 * priv_eps * priv_eps * l * (l + 1):
        #     print("log_t is smaller")
    else:
        log_t = priv_eps * l
    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)

def logmgf_exact_new(q, noise_eps, l, z, k, max_q):
    """Computes the logmgf value given q and privacy eps.
    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf by extending to k partitioning.
    The second term is from our paper.
    The third term comes directly from the privacy guarantee.
    Args:
        q: pr of non-optimal outcome
        gamma:
        l: moment to compute.
        t: the instance portion of each party
        z: the number of partitioning that may be affected in each party
        k: number of partitioning in each party
    Returns:
        Upper bound on logmgf
    """
    # k=1
    # z=1
    # k=z
    priv_eps = 2 * k * noise_eps
    # z=1
    # print("z:", z)
    if q < max_q:
    # if q < 0.5:
        part1 = (1-q) * math.pow((1-q) / (1-math.exp(2*z*noise_eps)*q), l)
        part2 = q * math.exp(2*z*noise_eps*l)
        try:
            log_t = math.log(part1 + part2)
        except ValueError:
            print("part1: ", part1)
            print("part2: ", part2)
            print("Got ValueError in math.log for values :" + str((q, noise_eps, l, t, z, k)))
            log_t = priv_eps * l
        # print("log_t:", log_t)
        # if log_t < 0.5 * priv_eps * priv_eps * l * (l + 1):
        #     print("log_t is smaller")
    else:
        log_t = priv_eps * l
    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


def logmgf_exact_party_level(q, noise_eps, l, k, max_q):
    """Computes the logmgf value given q and privacy eps.
    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf by extending to k partitioning.
    The second term is from our paper.
    The third term comes directly from the privacy guarantee.
    Args:
        q: pr of non-optimal outcome
        gamma:
        l: moment to compute.
        t: the instance portion of each party
        z: the number of partitioning that may be affected in each party
        k: number of partitioning in each party
    Returns:
        Upper bound on logmgf
    """
    priv_eps = 2 * k * noise_eps
    # print("priv eps:", priv_eps)
    # print("q:", q)
    # max_q=0.5
    if q < max_q:
        part1 = (1-q) * math.pow((1-q) / (1 - math.exp(priv_eps) * q), l)
        part2 = q * math.exp(priv_eps * l)
        try:
            log_t = math.log(part1 + part2)
        except ValueError:
            print("Party level Got ValueError in math.log for values :" + str((q, noise_eps, l, k)))
            log_t = priv_eps * l
        # print("log_t:", log_t)
        #if log_t < 0.5 * priv_eps * priv_eps * l * (l + 1):
        #    print("log_t is smaller")
    else:
        log_t = priv_eps * l
    # print("lot_t:", log_t)
    # print("min: ", min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l))
    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


  # if q < 0.5:
  #   t_one = (1-q) * math.pow((1-q) / (1 - math.exp(priv_eps) * q), l)
  #   t_two = q * math.exp(priv_eps * l)
  #   t = t_one + t_two
  #   try:
  #     log_t = math.log(t)
  #   except ValueError:
  #     print("Got ValueError in math.log for values :" + str((q, priv_eps, l, t)))
  #     log_t = priv_eps * l
  # else:
  #   log_t = priv_eps * l
  #
  # return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


# return the moment
def logmgf_from_counts(counts, noise_eps, l, z, k, max_q):
    """
    ReportNoisyMax mechanism with noise_eps with 2*noise_eps-DP
    in our setting where one count can go up by one and another
    can go down by 1.
    """
    q = compute_q_noisy_max(counts, noise_eps)
    # print("q:", q)
    # return logmgf_exact(q, noise_eps, l, t, z, k, max_q)
    return logmgf_exact_new(q, noise_eps, l, z, k, max_q)

def logmgf_from_counts_party_level(counts, noise_eps, l, k, max_q):
    """
    ReportNoisyMax mechanism with noise_eps with 2*noise_eps-DP
    in our setting where one count can go up by one and another
    can go down by 1.
    """
    q = compute_q_noisy_max(counts, noise_eps)
    return logmgf_exact_party_level(q, noise_eps, l, k, max_q)

def sens_at_k(counts, noise_eps, l, k):
    """Return sensitivity at distane k.
    Args:
    counts: an array of scores
    noise_eps: noise parameter used
    l: moment whose sensitivity is being computed
    k: distance
    Returns:
    sensitivity: at distance k
    """
    counts_sorted = sorted(counts, reverse=True)
    if 0.5 * noise_eps * l > 1:
        print("l too large to compute sensitivity")
        return 0
    # Now we can assume that at k, gap remains positive
    # or we have reached the point where logmgf_exact is
    # determined by the first term and ind of q.
    if counts[0] < counts[1] + k:
        return 0
    counts_sorted[0] -= k
    counts_sorted[1] += k
    val = logmgf_from_counts(counts_sorted, noise_eps, l, 0, 0, 0.5)
    counts_sorted[0] -= 1
    counts_sorted[1] += 1
    val_changed = logmgf_from_counts(counts_sorted, noise_eps, l, 0, 0, 0.5)
    return val_changed - val


def smoothed_sens(counts, noise_eps, l, beta):
    """Compute beta-smooth sensitivity.
    Args:
    counts: array of scors
    noise_eps: noise parameter
    l: moment of interest
    beta: smoothness parameter
    Returns:
    smooth_sensitivity: a beta smooth upper bound
    """
    k = 0
    smoothed_sensitivity = sens_at_k(counts, noise_eps, l, k)
    while k < max(counts):
        k += 1
        sensitivity_at_k = sens_at_k(counts, noise_eps, l, k)
        smoothed_sensitivity = max(
            smoothed_sensitivity,
            math.exp(-beta * k) * sensitivity_at_k)
        if sensitivity_at_k == 0.0:
            break
    return smoothed_sensitivity



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_path', type=str, default='privacy.npy.npz', help='The file that store the counts etc')
    parser.add_argument('--moments', type=int, default=8, help='Number of moments')
    parser.add_argument('--noise_eps', type=float, default=0.1, help='Eps value for each call to noisymax')
    parser.add_argument('--delta', type=float, default=1e-5, help='Target value of delta')
    parser.add_argument('--n_partition', type=int, default=1, help='Number of partitioning')
    parser.add_argument('--max_z', type=int, default=1)
    parser.add_argument('--is_local', type=int, default=0)
    parser.add_argument('--n_parties', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    arrays = np.load(args.np_path)
    t_arr = arrays['arr_0']
    z_arr = arrays['arr_1']
    counts_mat = arrays['arr_2']
    print("z arr:", z_arr)
    n_instances = counts_mat.shape[0]
    print("n instances:", n_instances)
    print("counts:", counts_mat)
    indices = np.array(range(n_instances))
    l_list = 1.0 + np.array(range(args.moments))
    total_log_mgf_nm = np.array([0.0 for _ in l_list])
    total_log_mgf_partylevel = np.array([0.0 for _ in l_list])
    total_ss_nm = np.array([0.0 for _ in l_list])
    # total_ss_nm = np.array([0.0 for _ in l_list])
    noise_eps = args.noise_eps
    # max_z = args.max_z
    max_z = z_arr.max()
    # z_arr_new = np.zeros(10)
    if max_z != 0:
        max_q = (1-math.exp(-2*max_z*noise_eps)) / (math.exp(2*max_z*noise_eps)-math.exp(-2*max_z*noise_eps))
    else:
        max_q = 0
    # max_q = (1 - np.sum(t_arr * np.exp(-2 * z_arr * noise_eps))) / (
    #             np.sum(t_arr * np.exp(2 * z_arr * noise_eps)) - np.sum(t_arr * np.exp(-2 * z_arr * noise_eps)))

    if args.is_local:
        query_each_party = len(counts_mat) / args.n_parties
        for j in range(args.n_parties):
            indices = np.array(range(query_each_party*j, query_each_party*(j+1)))
            total_log_mgf_nm = np.array([0.0 for _ in l_list])
            for i in indices:
                total_log_mgf_nm += np.array(
                    [logmgf_from_counts(counts_mat[i], noise_eps, l, 1, 1, max_q)
                     for l in l_list])
                # total_log_mgf_partylevel += np.array(
                #     [logmgf_from_counts_party_level(counts_mat[i], noise_eps, l, args.n_partition, max_q)
                #      for l in l_list])

            print("total_log_mgf_nm:", total_log_mgf_nm)
            delta = args.delta
            # We want delta = exp(alpha - eps l).
            # Solving gives eps = (alpha - ln (delta))/l
            eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list
            eps_list_partylevel = (total_log_mgf_partylevel - math.log(delta)) / l_list
            # print("Epsilons (Noisy Max): " + str(eps_list_nm))
            # print("local Epsilon = " + str(min(eps_list_nm)) + ".")
            if j == 0:
                eps_min = min(eps_list_nm)
            else:
                eps_min = max(min(eps_list_nm), eps_min)

        print("Epsilon = " + str(eps_min) + ".")

    else:
        for i in indices:
            total_log_mgf_nm += np.array(
                [logmgf_from_counts(counts_mat[i], noise_eps, l, max_z, max_z, max_q)
                 for l in l_list])
            total_ss_nm += np.array(
                [smoothed_sens(counts_mat[i], noise_eps, l, args.beta)
                 for l in l_list])
            total_log_mgf_partylevel += np.array(
                [logmgf_from_counts_party_level(counts_mat[i], noise_eps, l, args.n_partition, max_q)
                 for l in l_list])

        delta = args.delta

        print("total_log_mgf_nm:", total_log_mgf_nm)

        # We want delta = exp(alpha - eps l).
        # Solving gives eps = (alpha - ln (delta))/l
        eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list
        eps_list_partylevel = (total_log_mgf_partylevel - math.log(delta)) / l_list
        print("Epsilons (Noisy Max): " + str(eps_list_nm))
        print("Epsilon = " + str(min(eps_list_nm)) + ".")
        print("Epsilons (Noisy Max): " + str(eps_list_partylevel))
        print("Party Level Epsilon:" + str(min(eps_list_partylevel)) + ".")
        if min(eps_list_nm) == eps_list_nm[-1]:
            print("Warning: May not have used enough values of l")

        print("Smoothed sensitivities (Noisy Max): " + str(total_ss_nm / l_list))

        # If beta < eps / 2 ln (1/delta), then adding noise Lap(1) * 2 SS/eps
        # is eps,delta DP
        # Also if beta < eps / 2(gamma +1), then adding noise 2(gamma+1) SS eta / eps
        # where eta has density proportional to 1 / (1+|z|^gamma) is eps-DP
        # Both from Corolloary 2.4 in
        # http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
        # Print the first one's scale
        ss_eps = 2.0 * args.beta * math.log(1 / delta)
        ss_scale = 2.0 / ss_eps
        print("To get an " + str(ss_eps) + "-DP estimate of epsilon, ")
        print("..add noise ~ " + str(ss_scale))
        print("... times " + str(total_ss_nm / l_list))
        print("Epsilon = " + str(min(eps_list_nm)) + ".")
        if min(eps_list_nm) == eps_list_nm[-1]:
            print("Warning: May not have used enough values of l")

        # Data independent bound, as mechanism is
        # 2*noise_eps DP.
        data_ind_log_mgf = np.array([0.0 for _ in l_list])
        data_ind_log_mgf += n_instances * np.array(
            [logmgf_exact_new(1.0, 2.0 * noise_eps, l, 0, 0, max_q) for l in l_list])

        data_ind_eps_list = (data_ind_log_mgf - math.log(delta)) / l_list
        print("Data independent bound = " + str(min(data_ind_eps_list)) + ".")
    print("max_q:", max_q)
