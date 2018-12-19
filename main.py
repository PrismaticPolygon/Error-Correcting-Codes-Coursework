import numpy as np


class TransmissionFailure(Exception):

    pass


def hammingGeneratorMatrix(r):

    """input: a number r
    output: G, the generator matrix of the (2^r-1,2^r-r-1) Hamming code"""
    n = 2 ** r - 1

    # construct permutation pi
    pi = []
    for i in range(r):
        pi.append(2 ** (r - i - 1))
    for j in range(1, r):
        for k in range(2 ** j + 1, 2 ** (j + 1)):
            pi.append(k)

    # construct rho = pi^(-1)
    rho = []
    for i in range(n):
        rho.append(pi.index(i + 1))

    # construct H'
    H = []
    for i in range(r, n):
        H.append(decimalToVector(pi[i], r))

    # construct G'
    GG = [list(i) for i in zip(*H)]
    for i in range(n - r):
        GG.append(decimalToVector(2 ** (n - r - i - 1), n - r))

    # apply rho to get Gtranpose
    G = []
    for i in range(n):
        G.append(GG[rho[i]])

    # transpose
    G = [list(i) for i in zip(*G)]

    return G


def decimalToVector(n, r):

    """input: numbers n and r (0 <= n<2**r)
    output: a string v of r bits representing n"""

    v = []
    for s in range(r):
        v.insert(0, n % 2)
        n //= 2
    return v


def simulation(r, N, p):

    successes, failures, errors = 0, 0, 0

    for n in range(1, N + 1):

        try:

            print("\n*** Experiment {} of {} ***\n\n* Source *".format(n, N))

            m = randomMessage(r)

            print("Message\nm = {}".format(m))

            c = encoder(m)

            print("\nCodeword\nc = {}\n\n* Channel *".format(c))

            v = BSC(c, p)

            print("Received vector\nv = {}\n\n * Destination *".format(v))

            hatc = syndrome(v)

            print("\nCodeword estimate\nhatc = {}".format(hatc))

            hatm = retrieveMessage(hatc)

            print("\nMessage estimate\nhatm = {}".format(hatm))

            if list(m) == list(hatm):

                successes += 1

                print('\nDecoding success')

            else:

                errors += 1

                print('\nDecoding error')

        except TransmissionFailure:

            failures +=1

            print('\nDecoding failure')

    print("\n*** End of experiments ***\n")

    print("Successes: {}".format(successes))
    print("Failures: {}".format(failures))
    print("Errors: {}".format(errors))

    print("\nExperimental DEP: {}".format(errors / N))


def randomMessage(r):
    """Generate a message m of the right length uniformly at random"""

    return np.random.randint(2, size=2 ** r - r - 1)


def encoder(m):
    """Encode m with the extended Hamming code of redundancy r (calculated within) to get codeword c."""

    r = 0

    while (2 ** r) - r - 1 is not len(m):

        r += 1

    G = hammingGeneratorMatrix(r)

    c = np.asarray(np.dot(m, G) % 2).reshape(-1)

    return np.append(c, np.sum(c) % 2)  # Add parity-check bit.


def BSC(c, p):
    """Send c through a Binary Symmetric Channel with crossover probability p; the output is v."""

    return np.fromiter(map(lambda x: (x + 1) % 2 if np.random.rand() <= p else x, c), dtype=np.int8)


def syndrome(v):

    print("Decoding by syndrome")

    def parityCheckMatrix(r):

        matrix = np.matrix([decimalToVector((2 ** r) - 1, r)]).T

        for n in range((2 ** r) - 2, 0, -1):
            num = decimalToVector(n, r)

            num.reverse()

            matrix = np.insert(matrix, 0, num, axis=1)

        return matrix

    r = int(np.log2(len(v)))
    parity_violated = np.sum(v) % 2 != 0
    v = v[:-1]  # Remove parity bit

    syndrome = np.asarray(np.dot(v, parityCheckMatrix(r).T) % 2).reshape(-1)  # Calculate syndrome and convert to vector

    print("Syndrome: {}".format(syndrome))

    if np.sum(syndrome) == 0 and parity_violated:   # The parity bit has been flipped, but we don't care!

        return v

    elif np.sum(syndrome) == 0 and not parity_violated: # Nothing untoward has happened.

        return v

    elif np.sum(syndrome) != 0 and parity_violated:     # One, three, or five errors...

        i = np.sum([x * (2 ** i) for i, x in enumerate(syndrome)])

        print("Syndrome: {}".format(i + 2 ** r))

        print("i = {}".format(i))

        v[i - 1] = (v[i - 1] + 1) % 2

        return v

    elif np.sum(syndrome) != 0 and not parity_violated:  # Even number errors. Can't be fixed.

        raise TransmissionFailure


def retrieveMessage(c):
    """Retrieve an estimate of the message hatm from hatc"""

    r = int(np.log2(len(c) + 1))

    return np.delete(c, [(1 * 2 ** (n - 1)) - 1 for n in range(r, 0, -1)])
