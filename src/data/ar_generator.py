import statsmodels.tsa.api as smt
from drift_detection.entropy import *


def gen_exper_data1():
    #arparams = np.array([1.5, -0.4, -0.3, 0.2])
    arparams = np.array([0.9, -0.2, 0.8, -0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_1 = smt.arma_generate_sample(ar, ma, 2000)

    #arparams = np.array([-0.1, 1.2, 0.4, -0.5])
    arparams = np.array([-0.3, 1.4, 0.4, -0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_2 = smt.arma_generate_sample(ar, ma, 2000)

    y = np.concatenate((ar_1, ar_2))

    plt.plot(range(len(y)), y)
    plt.plot(range(len(ar_1), len(ar_1) + len(ar_2)), ar_2, c='r')
    plt.show()



def gen_exper_data2():
    arparams = np.array([0.9, -0.2, 0.8, -0.5])
    #arparams = np.array([1.1, -0.6, 0.8, -0.5, -0.1, 0.3])
    #arparams = np.array([0.5, 0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_1 = smt.arma_generate_sample(ar, ma, 3000, sigma=0.5)

    arparams = np.array([-0.3, 1.4, 0.4, -0.5])
    #arparams = np.array([-0.1, 1.2, 0.4, 0.3, -0.2, -0.6])
    #arparams = np.array([1.5, -0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_2 = smt.arma_generate_sample(ar, ma, 3000, sigma=1.5)


    arparams = np.array([1.5, -0.4, -0.3, 0.2])
    #arparams = np.array([1.2, -0.4, -0.3, 0.7, -0.6, 0.4])
    # arparams = np.array([0.9, -0.2, 0.8, -0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_3 = smt.arma_generate_sample(ar, ma, 3000, sigma=2.5)

    arparams = np.array([-0.1, 1.4, 0.4, -0.7])
    #arparams = np.array([-0.1, 1.1, 0.5, 0.2, -0.2, -0.5])
    #arparams = np.array([0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4])

    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_4 = smt.arma_generate_sample(ar, ma, 3000, sigma=3.5)

    y = np.concatenate((ar_1, ar_2, ar_3, ar_4))

    # plt.plot(range(len(y)), y)
    # plt.show()



    #arparams = np.array([0.9, -0.2, 0.8, -0.5])
    #arparams = np.array([1.1, -0.6, 0.8, -0.5, -0.1, 0.3])
    arparams = np.array([0.5, 0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_1 = smt.arma_generate_sample(ar, ma, 3000, sigma=0.5)

    #arparams = np.array([-0.3, 1.4, 0.4, -0.5])
    #arparams = np.array([-0.1, 1.2, 0.4, 0.3, -0.2, -0.6])
    arparams = np.array([1.5, -0.5])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_2 = smt.arma_generate_sample(ar, ma, 3000, sigma=1.5)

    arparams = np.array([0.9, -0.2, 0.8, -0.5])
    #arparams = np.array([1.5, -0.4, -0.3, 0.2])
    #arparams = np.array([1.2, -0.4, -0.3, 0.7, -0.6, 0.4])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_3 = smt.arma_generate_sample(ar, ma, 3000, sigma=2.5)

    #arparams = np.array([-0.1, 1.4, 0.4, -0.7])
    #arparams = np.array([-0.1, 1.1, 0.5, 0.2, -0.2, -0.5])
    arparams = np.array([0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4])
    maparams = np.array([0])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ar_4 = smt.arma_generate_sample(ar, ma, 3000, sigma=3.5)

    y = np.concatenate((ar_1, ar_2, ar_3, ar_4))

    plt.plot(range(len(y)), y)
    plt.show()



if __name__ == '__main__':
    gen_exper_data1()

