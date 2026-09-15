import numpy as np
from moocore import Hypervolume

class D_PHI():
    """Calculate the D-PHI and complementary indicator (CI) values for a given solution set.

        Parameters (The default is to minimize a problem):
        y : np.ndarray
            A set of solutions (objective vectors).

        aspirationp : 1-D np.ndarray
            An aspiration point (objective vector) consists of aspiration levels ai.

        reservationp : 1-D np.ndarray
            A reservation point (objective vector) consists of reservation levels ri.

        c1 : float or 1-D np.ndarray, optional
            Desirability values when yi=ai, c1 = DF(ai). Defaults to 1 for all objectives.

        c2 : float or 1-D np.ndarray, optional
            Desirability values when yi=ri, c2 = DF(ri). Defaults to 0 for all objectives.

        epsilon_a: float or 1-D np.ndarray, optional
            A positive parameter used to define the upper limit of the desirability function,
            influencing the extent to which high-quality solutions (better than aspiration levels) are rewarded.
            Defaults to 0.2, then the upper limit is c1+epsilon_a = 1.2.

        epsilon_r: float or 1-D np.ndarray, optional
            A positive parameter used to define the lower limit of the desirability function,
            influencing the extent to which low-quality solutions (worse than reservation levels) are penalized.
            Defaults to 0.2, then the lower limit is c2-epsilon_r = -0.2.

        delta : float, optional
            A small positive value used when ai == ri. In this case, ai - ri is set to -delta
            (if ai=ri, ai=ri-delta. That is, delta=ri-ai, when ai=ri.)


        Returns:
        D-PHI and CI values
            Two float numbers.


        A 2-D Example:
            aspirationp = np.array([0.1, -0.5])
            reservationp = np.array([0.3, 0])
            indicators = D_PHI(solutions, aspirationp, reservationp)
            # transferred_solutions = indicators.df()
            dphi, ci = indicators.get_values()

        """

    def __init__(self, y, aspirationp, reservationp, c1=1, c2=0, epsilon_a=0.2, epsilon_r=0.2, delta=1e-2):
        """Initialize with an aspiration point and a reservation point.
        """
        ai = np.array([aspirationp]).flatten()  # Make sure ai/ri is a 1-D np array
        ri = np.array([reservationp]).flatten()

        c1 = np.array([c1]).flatten()  # Make sure c1/c2 is a 1-D np array
        c2 = np.array([c2]).flatten()

        epsilon_a = np.array([epsilon_a]).flatten()  # Make sure epsilon_a/epsilon_r is a 1-D np array
        epsilon_r = np.array([epsilon_r]).flatten()

        y = np.atleast_2d(y)  # Ensure that the calculation can be performed even when there is only one solution in the set

        if len(ai) != len(ri):
            raise ValueError("aspiration point and reservation point should have the same length")

        if delta <= 0:
            raise ValueError("delta should be a small value greater than 0")

        if len(c1) != len(c2):
            raise ValueError("c1 and c2 should have the same length")

        if len(epsilon_a) != len(epsilon_r):
            raise ValueError("epsilon_a and epsilon_r should have the same length")

        num, dim = y.shape

        self.aspirationp = ai
        self.reservationp = ri
        self.delta = -delta
        self.c1 = c1
        self.c2 = c2
        self.epsilon_a = epsilon_a
        self.epsilon_r = epsilon_r
        self.y = y

        if len(self.aspirationp) != 1 and len(self.aspirationp) != dim:
            raise ValueError("The length of aspiration/reservation point should be 1 or the same as the dimension of y")

        if len(self.aspirationp) == 1 and len(self.aspirationp) < dim:
            # If set the same aspiration and reservation levels for all objectives
            # Expand aspirationp and reservationp so that their lengths are the same as the dimension of y.
            self.aspirationp = np.full(dim, self.aspirationp)
            self.reservationp = np.full(dim, self.reservationp)

        if not np.all(self.aspirationp <= self.reservationp):
            raise ValueError("All aspiration levels should be smaller than or equal to the corresponding reservation levels")

        if len(self.c1) != 1 and len(self.c1) != dim:
            raise ValueError("The length of c1/c2 should be 1 or the same as the dimension of y")

        if len(self.c1) == 1 and len(self.c1) < dim:
            # Expand c1 and c2 so that their lengths are the same as the dimension of y
            self.c1 = np.full(dim, self.c1)
            self.c2 = np.full(dim, self.c2)

        if not np.all(self.c1 > self.c2):
            raise ValueError("The desirability value of the aspiration level should be greater than the desirability value of the reservation level")

        if len(self.epsilon_a) != 1 and len(self.epsilon_a) != dim:
            raise ValueError("The length of epsilon_a/epsilon_r should be 1 or the same as the dimension of y")

        if len(self.epsilon_a) == 1 and len(self.epsilon_a) < dim:
            # Expand epsilon_a and epsilon_r so that their lengths are the same as the dimension of y
            self.epsilon_a = np.full(dim, self.epsilon_a)
            self.epsilon_r = np.full(dim, self.epsilon_r)


        if np.any(self.epsilon_a <= 0) or np.any(self.epsilon_r <= 0):
            raise ValueError("Epsilon_a and epsilon_r should be positive numbers")

        self.a_minus_r = self.aspirationp - self.reservationp
        # ai should be smaller than ri, ensuring that ai - ri is less than 0
        # If ai == ri, set ai = ri - delta, so that ai - ri = -delta (self.delta) < 0
        self.a_minus_r = np.where(self.a_minus_r < 0, self.a_minus_r, self.delta)


    def less_than_ai(self, y, a, r, a_minus_r, c1, c2, epsilon_a=0.2):
        d = (-epsilon_a ** 2 * (a_minus_r) / (c1 - c2)) * (1 / (y - a + epsilon_a * (a_minus_r) / (c1 - c2))) + (c1 + epsilon_a)
        return d

    def between_r_a(self, y, a, r, a_minus_r, c1, c2):
        d = ((c1 - c2) / (a_minus_r)) * y + (-c1 * r + c2 * a) / (a_minus_r)
        return d

    def greater_than_ri(self, y, a, r, a_minus_r, c1, c2, epsilon_r=0.2):
        d = (-epsilon_r ** 2 * (a_minus_r) / (c1 - c2)) * (1 / (y - r - epsilon_r * (a_minus_r) / (c1 - c2))) + (c2 - epsilon_r)
        return d

    def df(self):  # Calculate the desirability function value of all solutions
        y = self.y
        num, dim = y.shape

        d = y.copy().astype(float)
        for i in range(dim):
            # Calculate desirability function values for solutions on each objective function
            comp = np.zeros(num)
            currenty = y[:, i]
            currentd = currenty.copy().astype(float)
            comp1 = currenty > self.reservationp[i]
            comp2 = currenty <= self.aspirationp[i]
            comp = comp1.astype(int) - comp2.astype(int)  # comp=-1 for y<=ai; comp=0 for y∈(ai,ri]; comp=1 for y>ri
            currentd[comp == -1] = self.less_than_ai(currentd[comp == -1], self.aspirationp[i], self.reservationp[i], self.a_minus_r[i], self.c1[i], self.c2[i], self.epsilon_a[i])
            currentd[comp == 0] = self.between_r_a(currentd[comp == 0], self.aspirationp[i], self.reservationp[i], self.a_minus_r[i], self.c1[i], self.c2[i])
            currentd[comp == 1] = self.greater_than_ri(currentd[comp == 1], self.aspirationp[i], self.reservationp[i], self.a_minus_r[i], self.c1[i], self.c2[i], self.epsilon_r[i])
            d[:, i] = currentd

        return d

    def ASF_value(self):  # Define achievement scalarization function
        ai = self.aspirationp
        ri = self.reservationp
        ri_minus_ai = - self.a_minus_r
        y = self.y

        # reference point
        z_star = ai

        # weight
        lambda_w = 1 / (ri_minus_ai)

        t = lambda_w * (y- z_star)

        asf = np.max(t, axis=1)

        return asf

    def get_values(self):  # Calculate the D-PHI and CI values of the solution set
        # Set a reference point to calculate HV values
        reference_point = self.c2 - self.epsilon_r

        df_objs = self.df() # Calculate the desirability function value of all solutions (transfer solutions)

        hv = Hypervolume(ref=reference_point, maximise=True)
        dphi = hv(df_objs)  # D-PHI value

        asf = self.ASF_value()
        index_asf = np.argmin(asf)
        ci = -np.min(asf)  # CI (complementary indicator) value

        return dphi, ci
