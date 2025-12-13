import matplotlib.pyplot as plt

def plot_reward_terms(terms_list):
    """
    terms_list: list of dict
    e.g. [{"total": 0.9, "r_dist": 0.01, ...}, {...}]
    """

    # çizmek istediğimiz terimler (sende olmayan varsa otomatik 0 basar)
    keys = [
        "total",
        "r_dist",
        "r_safety",
        "r_eff",
        "p_jerk",
        "p_att",
        "p_time",
        "bonus",
        "p_danger"
    ]

    for k in keys:
        series = [float(t.get(k, 0.0)) for t in terms_list]

        plt.figure()
        plt.plot(series)
        plt.title(k)
        plt.xlabel("step")
        plt.ylabel(k)
        plt.grid(True)

    plt.show()
