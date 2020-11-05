import cmath
import numpy as np

from math import pi

from .invert import SurroundVariation, reduce
from .sld_profile import SLDProfile, refr_idx

try:  # CRUFT: basestring isn't used in python3
    basestring
except:
    basestring = str

ZERO_TOL = 1e-10
# The tolerance to decide, when the reflectivity is 1, i.e. |r - 1| < tol.
REFLECTIVITY_UNITY_TOL = 1e-10

"""
    References:
    
    [Majkrzak2003] C. F. Majkrzak, N. F. Berk and U. A. Perez-Salas. Langmuir (2003), 19, 7796-7810.
    Phase-Sensitive Neutron Reflectometry

"""

class AbstractReferenceVariation(SurroundVariation):
    def __init__(self, fronting_sld, backing_sld):
        """
        fronting and backing sld are in units of [AA^-2].

        Example:
            For a backing SLD of Si, use backing_sld = 2.1e-6

        :param fronting_sld:
        :param backing_sld:
        """
        self._f = float(fronting_sld)
        self._b = float(backing_sld)
        self._measurements = []

        self.dImagR = None
        self.dRealR = None

        self.plot_imaginary_offset = -5

    def run(self):
        """
        Runs the phase reconstruction algorithm on the loaded data set.

        The result (Real and Imaginary part of the reflection) can be access by the attributes
            self.Q, self.RealR, self.ImagR

        :return: None
        """
        self._check_measurements()
        self._calc()
        self.Qin = self.Q

    def _check_measurements(self):
        """
        Checks that the given measurements coincide on the q-grid.
        Checks that the given SLDs are not equal (breaks the reconstruction algorithm)

        :raise: RuntimeWarning if the checks fail
        :return: None
        """
        if len(self._measurements) <= 1:
            return

        q1 = self._measurements[0]['Qin']

        for ms in self._measurements:
            if not ms['Qin'].shape == q1.shape or not all(q1 == ms['Qin']):
                raise RuntimeWarning("Q points do not match in data files")

        slds = [ms['sld'] for ms in self._measurements]

        # Checks that there are no duplicate SLDs added
        # duplicate sld profile yield a singular matrix in the constraint system (obviously)
        if any([x == y for i, x in enumerate(slds) for j, y in enumerate(slds) if i != j]):
            raise RuntimeWarning("Two equal sld profiles found. The profiles have to be "
                                 "different.")

    def remesh(self, interpolation=1, interpolation_kind=None):
        """
        Re-meshes the loaded data onto a common grid by interpolation.

        Usually, the reflectivity data is not measured at the very same q points. Also the min/max range of
        the q values might vary. But, to reconstruct the phase, the reflectivity needs to be measured at
        the same q values. This method achieves this goal.

        The new grid is the coarsest possible grid. The min/max values are chosen such that every reflectivity
        measurement contains the min/max values.

        :param interpolation: integer number. Defines the number of additional interpolations between
            two q-grid points
        :param interpolation_kind: interpolation of the function between the point (linear/quadratic/etc..)
            See scipy.interp1d for possible values
        :return: None
        """

        # coarsest possible grid
        qmin = max([ms['Qin'][0] for ms in self._measurements])
        qmax = min([ms['Qin'][-1] for ms in self._measurements])
        npts = min([len(ms['Qin']) for ms in self._measurements])

        new_mesh = np.linspace(qmin, qmax, npts + 1)

        for measurement in self._measurements:
            q, R, dR = measurement['Qin'], measurement['Rin'], measurement['dRin']
            try:
                from skipi.function import Function
                f = Function.to_function(q, R, interpolation=interpolation_kind).remesh(new_mesh).oversample(
                    interpolation)

                if dR is not None:
                    df = Function.to_function(q, dR, interpolation=interpolation_kind).remesh(
                        new_mesh).oversample(interpolation)
                    dR = df.eval()

                measurement['Qin'], measurement['Rin'], measurement['dRin'] = f.get_domain(), f.eval(), dR
            except:
                # Fallback if skipi is not available
                q, R = remesh([q, R], qmin, qmax, npts, left=0, right=0)
                if dR is not None:
                    q, dR = remesh([q, dR], qmin, qmax, npts, left=0, right=0)

                measurement['Qin'], measurement['Rin'], measurement['dRin'] = q, R, dR

    def load_data(self, q, R, dq, dR, sld_profile, name='unknown'):
        """
        Load data directly

        :param q: array of q
        :param R: array of R(q)
        :param dq: array of error in q or None
        :param dR: array of error in R(q) or None
        :param sld_profile: reference layer as SLDProfile
        :param name: name of the measurement
        :return: None
        """
        assert isinstance(sld_profile, SLDProfile)

        self._measurements.append(
            {'name': name, 'Qin': q, 'dQin': dq, 'Rin': R, 'dRin': dR, 'sld': sld_profile})

    def load_function(self, function, sld_profile, name='unknown'):
        """
        Load the reflectivity data by using the skipi function package.

        :param function: reflectivity measurement, including errors in dx, dy
        :param sld_profile: reference layer as SLDProfile
        :param name: name of the measurement
        :return: None
        """
        from skipi.function import Function

        assert isinstance(sld_profile, SLDProfile)
        assert isinstance(function, Function)

        self._measurements.append({
            'name': name,
            'Qin': function.get_domain(),
            'dQin': function.dx.eval(),
            'Rin': function.eval(),
            'dRin': function.dy.eval(),
            'sld': sld_profile
        })

    def load(self, file, sld_profile, use_columns=None, name=None, q0=0):
        """
        Load the reflectivity data by reading from a file.

        File structure:
            q, R(q), [dR(q)] for 2-3 column files
            q, dq, R(q), dR(q), [wavelength] for 4-5 column files

        :param file: file name
        :param sld_profile: reference layer as SLDProfile
        :param use_columns: columns to read the data from
        :param name: optional name of the measurement, if None then filename is used
        :param q0: minimum q value to consider, all measurements below q0 are discarded
        :return: None
        """
        assert isinstance(sld_profile, SLDProfile)

        if isinstance(file, basestring):
            d = np.loadtxt(file, usecols=use_columns).T
            _name = file
        else:
            d = file
            _name = "Measurement {}".format(len(self._measurements) + 1)

        if name is None:
            name = _name

        q, dq, r, dr = None, None, None, None

        ncols = len(d)
        if ncols <= 1:
            raise ValueError("Data file has less than two columns")
        elif ncols == 2:
            q, r = d[0:2]
            dr = None
        elif ncols == 3:
            q, r, dr = d[0:3]
            dq = None
        elif ncols == 4:
            q, dq, r, dr = d[0:4]
        elif ncols >= 5:
            q, dq, r, dr, lamb = d[0:5]

        dq = dq[q > q0] if dq is not None else None
        dr = dr[q > q0] if dr is not None else np.zeros(len(r))
        r = r[q > q0]
        q = q[q > q0]

        self._measurements.append(
            {'name': name, 'Qin': q, 'dQin': dq, 'Rin': r, 'dRin': dr, 'sld': sld_profile})

    def _calc(self):
        self.Q, self.Rall, self.dR = self._phase_reconstruction()
        l = len(self.Rall)

        # If only two measurements are supplied, Rall contains then
        # two branches of the reflection. This code splits the tuple up into R+ and R-
        self.Rp, self.Rm = np.zeros(l, dtype=complex), np.zeros(l, dtype=complex)
        for idx, el in enumerate(self.Rall):
            if type(el) is np.ndarray or type(el) is list:
                self.Rp[idx] = el[0]
                self.Rm[idx] = el[1]
            else:
                self.Rp[idx] = el
                self.Rm[idx] = el

        # default selection
        self.R = self.Rp
        self.RealR, self.ImagR = self.R.real, self.R.imag

    @classmethod
    def _refl(cls, alpha_u, beta_u, gamma_u):
        # Compute the reflection coefficient, based on the knowledge of alpha_u, beta_u,
        # gamma_u where these parameters are the solution of the matrix equation
        #
        # See eq (38)-(40) in [Majkrzak2003]
        return - (alpha_u - beta_u + 2 * 1j * gamma_u) / (alpha_u + beta_u + 2)

    @classmethod
    def _drefl(cls, alpha, beta, gamma, cov):
        r"""
        Estimates the error within the reconstructed phase information (real and imaginary part are
        calculated separately) by the propagation of error method of the formula:

        ..math::
            R = (\beta - \alpha - 2i\gamma) / (\alpha + \beta + 2)

        :param alpha: alpha_u: solution of the linear equation
        :param beta: beta_u: solution of the linear equation
        :param gamma: gamma_u: solution of the linear equation
        :param cov: Covariance matrix of the linear least squares method, with the order
            [alpha, beta, gamma] = [0, 1, 2]
        :return: Estimated error as complex number
        """
        a, b, g = alpha, beta, gamma

        # dR/d alpha, real part
        dAlpha = - 2 * (b + 1) / (a + b + 2) ** 2
        # dR/d beta, real part
        dBeta = 2 * (a + 1) / (a + b + 2) ** 2
        # dR/d gamma = 0 (real part is independent of gamma)

        # dR/d alpha, imag part
        dAlphaIm = -2 * g / (a + b + 2) ** 2
        # dR/d beta, imag part
        dBetaIm = -2 * g / (a + b + 2) ** 2
        # dR/d gamma, imag part
        dGammaIm = -2 / (a + b + 2)

        r"""
        Error propagation
        
        ..math::
        Re(dR)^2 = (\frac{dR}{d\alpha} \sigma_\alpha)^2 + (\frac{dR}{d\beta} \sigma_\beta)^2 +
            2 * \frac{dR}{d\alpha}{\frac{dR}{d\beta} \sigma_{\alpha, \beta}
            
        where :math:`\sigma_\alpha` denotes the error within :math:`\alpha` and :math:`\sigma_{\alpha, \beta}`
        denotes the covariance of :math:`\alpha' and :math:`\beta`
        
        The error for the imaginary part is computed analogously.
        """
        sResqr = dAlpha ** 2 * cov[0][0] + dBeta ** 2 * cov[1][1] + 2 * dAlpha * dBeta * cov[0][1]
        sImsqr = dAlphaIm ** 2 * cov[0][0] + dBetaIm ** 2 * cov[1][1] + dGammaIm ** 2 * cov[2][2] + \
                 2 * dAlphaIm * dBetaIm * cov[0][1] + \
                 2 * dAlphaIm * dGammaIm * cov[0][2] + \
                 2 * dBetaIm * dGammaIm * cov[1][2]

        return np.sqrt(sResqr) + 1j * np.sqrt(sImsqr)

    @classmethod
    def _calc_refl_constraint(cls, q, reflectivity, sld_reference, fronting, backing):
        # See the child classes for implementations
        raise NotImplementedError()

    def _do_phase_reconstruction(self, q, Rs, dRs, SLDs):
        """
        The calculation is split up in multiple parts (to keep the code repetition low).
        First, we calculate the constraining linear equations for the reflection coefficient.
        Only this depends on the location of the reference layer (front or back). Next,
        we solve this linear system to retrieve some coefficients for the reflection
        coefficient. The linear system needs at least two measurements (yielding two
        reflection coefficients). Using the solution of the linear system, we finally
        calculate the reflection coefficient.

        Note that this reconstructs the reflection and also returns a new q value since this might have
        changed do to a non-zero fronting medium.

        :param q: The q value
        :param Rs: The reflections measured at q
        :param dRs: The uncertainties in the measured reflections at q
        :param SLDs: The SLDs corresponding to the reflections
        :return: q_new, Reflection, Error in Reflection
        """
        # Shift the q vector if the incidence medium is not vacuum
        # See eq (49) in [Majkrzak2003]
        #
        # Note that this also prohibits to measure the phase information
        # below the critical edge by simply flipping the sample.
        # The first q value you effectively measure is the first
        # one direct _after_ the critical edge ..
        q = cmath.sqrt(q ** 2 + 16.0 * pi * self._f).real

        # Skip those q values which are too close to zero, this would break the
        # refractive index calculation otherwise
        if abs(q) < ZERO_TOL:
            return None

        f = refr_idx(q, self._f)
        b = refr_idx(q, self._b)

        A = []
        c = []
        # Calculate for each measurement a linear constraint. Putting all of the
        # constraints together enables us to solve for the reflection itself. How to
        # calculate the linear constraint using a reference layer can be
        # found in [Majkrzak2003]

        for R, dR, SLD in zip(Rs, dRs, SLDs):
            # Don't use values close to the total refection regime.
            # You can't reconstruct the reflection below there with this method.
            if abs(R - 1) < REFLECTIVITY_UNITY_TOL:
                return None

            lhs, rhs, drhs = self._calc_refl_constraint(q, R, SLD, f, b)

            sigma = 1e-10

            # Note: the right hand side is a function of R and thus, the std deviation of the rhs is
            # simply the derivative times the std deviation of R
            if abs(dR) > ZERO_TOL:
                sigma = drhs * dR

            # divide by sigma, so that we do a chi squared minimization.
            A.append(np.array(lhs) / sigma)
            c.append(rhs / sigma)

        try:
            R, dR = self._solve_reference_layer(A, c)
            return q, R, dR
        except RuntimeWarning as e:
            print("Could not reconstruct the phase for q = {}. Reason: {}".format(q, e))

    def _phase_reconstruction(self):
        """
        Here, we reconstruct the reflection coefficients for every q.

        :return: q, r(q), dr(q) for each q
        """
        qr = np.empty(len(self._measurements[0]['Qin']), dtype=tuple)

        SLDs = [ms['sld'] for ms in self._measurements]

        for idx, q in enumerate(self._measurements[0]['Qin']):
            Rs = [ms['Rin'][idx] for ms in self._measurements]
            dRs = [ms['dRin'][idx] for ms in self._measurements]

            qr[idx] = self._do_phase_reconstruction(q, Rs, dRs, SLDs)

        qs, rs, dRs = zip(*qr[qr != None])

        return np.array(qs), np.array(rs), np.array(dRs)

    @classmethod
    def _solve_reference_layer(cls, A, c):
        """
        Solving the linear system A x = c
            with x = [alpha_u, beta_u, gamma_u], being the unknown coefficients for the
            reflection;
                    A being the lhs (except the x - variables)
                    c being the rhs
                    of the equation (38) in [Majkrzak2003]
            and returning the corresponding reflection coefficient calculated by alpha_u,
            .. gamma_u.

            A has to be a (Nx3) matrix, c has to be a (N)-vector (not checked)

            N <= 1:
                An exception is raised

            N == 2:
                The condition gamma^2 = alpha * beta - 1 will be used to construct two
                reflection coefficients which solve the equation. A list of two reflection
                coefficients is returned then.
            N >= 3:
                A least squares fit is performed (A^T A x = c)
                (which is exact for N=3)

            If any of the operations is not possible, (bad matrix condition number,
            quadratic eq has no real solution) a RuntimeWarning exception is raised
        """

        if len(A) <= 1:
            # Happens for q <= q_c, i.e. below the critical edge
            # Or the user has just specified one measurement ...
            raise RuntimeWarning("Not enough measurements to determine the reflection")

        if len(A) == 2:
            # Use the condition gamma^2 = alpha * beta - 1
            # First, calculate alpha, beta as a function of gamma,
            # i.e. alpha = u1 - v2*gamma, beta = u2 - v2*gamma
            B = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
            u = np.linalg.solve(B, c)
            v = np.linalg.solve(B, [A[0][2], A[1][2]])
            # Alternatively
            # Binv = np.linalg.inv(B)
            # u = np.dot(Binv, c)
            # v = np.dot(Binv, [A[0][2], A[1][2]])

            # Next, we can solve the equation gamma^2 = alpha * beta - 1
            # by simply substituting alpha and beta from above.
            # This then yields a quadratic equationwhich can be easily solved by:
            #   -b +- sqrt(b^2 - 4ac)  /  2a
            # with a, b, c defined as
            a = v[0] * v[1] - 1
            b = - (u[0] * v[1] + u[1] * v[0])
            c = u[0] * u[1] - 1
            # Notice, that a, b and c are symmetric (exchanging the rows 0 <-> 1)
            det = b ** 2 - 4 * a * c

            # This calculates alpha_u and beta_u.
            # Since they are "symmetric" in the sense alpha -> beta by switching
            # the order of the measurements, this can be done in this formula.
            # alpha = u - v * gamma, see above, the linear relationship
            alpha_beta = lambda u, v, g: u - v * g

            if abs(det) < ZERO_TOL:
                # Luckily, we get just one solution for gamma :D
                gamma_u = -b / (2 * a)
                alpha_u = alpha_beta(u[0], v[0], gamma_u)
                beta_u = alpha_beta(u[1], v[1], gamma_u)
                return cls._refl(alpha_u, beta_u, gamma_u), 0
            elif det > 0:
                reflection = []
                # Compute first gamma using both branches of the quadratic solution
                # Compute then alpha, beta using the linear dependence
                # Compute the reflection and append it to the solution list
                for sign in [+1, -1]:
                    gamma_u = (-b + sign * cmath.sqrt(det).real) / (2 * a)
                    alpha_u = alpha_beta(u[0], v[0], gamma_u)
                    beta_u = alpha_beta(u[1], v[1], gamma_u)
                    reflection.append(cls._refl(alpha_u, beta_u, gamma_u))

                # Returns the reflection branches, R+ and R-
                return reflection, 0
            else:
                # This usually happens if the reference sld's are not correct.
                raise RuntimeWarning("The quadratic equation has no real solution.")


        if len(A) >= 3:
            # least squares solves exact for 3x3 matrices
            # Silence the FutureWarning with rcond=None
            solution, residuals, rank, singular_values = np.linalg.lstsq(A, c, rcond=None)
            alpha_u, beta_u, gamma_u = solution

            # covariance matrix
            C = np.linalg.inv(np.array(A).T.dot(A))

            return cls._refl(alpha_u, beta_u, gamma_u), cls._drefl(alpha_u, beta_u, gamma_u, C)

    def choose(self, plus_or_minus):
        """
        If only two measurements were given, we calculated two possible reflection coefficients
        which have jumps, called R+ and R- branch.

        This method tries to selects from these two R's a continuously differentiable R,
        i.e. making a physically reasonable R. Note that, this R might not be the real
        reflection. From the two R's, we can join also two R's which are cont. diff'able. To
        select between them, use the plus_or_minus parameter (i.e. 0 or 1)

        :param plus_or_minus: int, being 0 or 1. 0 selects the R+ branch, 1 selects R-
        branch as the starting point
        :return:
        """
        pm = int(plus_or_minus) % 2

        r = [self.Rp.real, self.Rm.real]
        r_imag = [self.Rp.imag, self.Rm.imag]
        result = [r[pm % 2][0], r[pm % 2][1]]
        jump = []
        djump = []

        for idx in range(2, len(self.R)):

            c_next = result[idx - 1] - r[pm % 2][idx]
            c_nextj = result[idx - 1] - r[(pm + 1) % 2][idx]

            # Theoretically, we have an equidistant spacing in q, so, dividing is not necessary
            # but because sometimes we 'skipped' some q points (e.g. bad conditioning number),
            # it is crucial to consider this. At exactly these 'skipped' points, the selection
            # fails then
            dm_prev = (result[idx - 1] - result[idx - 2]) / (self.Q[idx - 1] - self.Q[idx - 2])
            dm_next = (r[pm % 2][idx] - result[idx - 1]) / (self.Q[idx] - self.Q[idx - 1])
            dm_nextj = (r[(pm + 1) % 2][idx] - result[idx - 1]) / (
                    self.Q[idx] - self.Q[idx - 1])

            continuity_condition = abs(c_next) > abs(c_nextj)
            derivative_condition = abs(dm_prev - dm_next) > abs(dm_prev - dm_nextj)

            # if you add more logic, be careful with pm = pm+1
            # with the current logic, it is not possible to have pm = pm + 2 (which does
            # nothing in fact, bc of mod 2)
            if continuity_condition and derivative_condition:
                jump.append(idx)
                pm = pm + 1
            elif derivative_condition:
                djump.append(idx)
                pm = pm + 1

            result.append(r[pm % 2][idx])

        pm = int(plus_or_minus) % 2
        imag_result = [r_imag[pm % 2][0], r_imag[pm % 2][1]]
        for idx in range(2, len(self.R)):
            if idx in jump or idx in djump:
                pm = pm + 1
            imag_result.append(r_imag[pm % 2][idx])

        return np.array(result) + 1j * np.array(imag_result), jump, djump

    def plot_r_branches(self):
        import pylab
        # fyi:  1e4 = 100^2
        pylab.plot(self.Q, 1e4 * self.Rp.real * self.Q ** 2, '.', label='Re R+')
        pylab.plot(self.Q, 1e4 * self.Rm.real * self.Q ** 2, '.', label='Re R-')

        pylab.plot(self.Q, 1e4 * self.Rm.imag * self.Q ** 2 + self.plot_imaginary_offset, '.',
                   label='Im R+')
        pylab.plot(self.Q, 1e4 * self.Rp.imag * self.Q ** 2 + self.plot_imaginary_offset, '.',
                   label='Im R-')

        pylab.xlabel("q")
        pylab.ylabel("(100 q)^2 R(q)")
        pylab.legend()

    def plot_r_choose(self, branch_selection=1, plot_jumps_continuity=True,
                      plot_jumps_derivative=True):
        import pylab
        r, jump, djump = self.choose(branch_selection)

        pylab.plot(self.Q, 1e4 * r.real * self.Q ** 2, '.', label='Re R')
        pylab.plot(self.Q, 1e4 * r.imag * self.Q ** 2 + self.plot_imaginary_offset, '.',
                   label='Im R')

        pylab.xlabel("q")
        pylab.ylabel("(100 q)^2 R(q)")

        if plot_jumps_continuity:
            label = 'Continuity jump'
            for j in jump:
                pylab.axvline(x=self.Q[j], color='red', label=label)
                # only plot the label once
                label = ''

        if plot_jumps_derivative:
            label = "Derivative jump"
            for j in djump:
                pylab.axvline(x=self.Q[j], color='black', label=label)
                # only plot the label once
                label = ''

        pylab.legend()


class BottomReferenceVariation(AbstractReferenceVariation):
    @classmethod
    def _calc_refl_constraint(cls, q, reflectivity, sld_reference, fronting,
                              backing):
        """
            Solving the linear system A x = c
            with x = [alpha_u, beta_u, gamma_u], being the unknown coefficients for the
            reflection;
                     A being the lhs (except the x - variables)
                     c being the rhs
                of the equation (38) in [Majkrzak2003]
            This method returns one row in this linear system as: lhs, rhs
            with lhs being one row in the matrix A and rhs being one scalar in the vector b
            drhs is the "variance" of the rhs, i.e. just the first derivative of rhs
        """
        w, x, y, z = sld_reference.as_matrix(q)
        f, b = fronting, backing

        alpha = (w ** 2 + 1 / (b ** 2) * y ** 2)
        beta = (b ** 2 * x ** 2 + z ** 2)
        gamma = (b * w * x + 1 / b * y * z)

        lhs = [f ** 2 * beta, b ** 2 * alpha, 2 * f * b * gamma]
        rhs = 2 * f * b * (1 + reflectivity) / (1 - reflectivity)
        drhs = 4 * f * b * 1 / (1 - reflectivity) ** 2

        return lhs, rhs, drhs


class TopReferenceVariation(AbstractReferenceVariation):
    @classmethod
    def _calc_refl_constraint(cls, q, reflectivity, sld_reference, fronting,
                              backing):
        """
            Solving the linear system A x = c
                with x = [alpha_u, beta_u, gamma_u], being the unknown coefficients for the
                reflection;
                     A being the lhs (except the x - variables)
                     c being the rhs
                of the equation (33) in [Majkrzak2003]
            This method returns one row in this linear system as: lhs, rhs
            with lhs being one row in the matrix A and rhs being one scalar in the vector b
            drhs is the "variance" of the rhs, i.e. just the first derivative of rhs
        """
        w, x, y, z = sld_reference.as_matrix(q)
        f, b = fronting, backing

        alpha = (1 / (f ** 2) * y ** 2 + z ** 2)
        beta = (f ** 2 * x ** 2 + w ** 2)
        gamma = (1 / f * w * y + f * x * z)

        lhs = [b ** 2 * beta, f ** 2 * alpha, 2 * f * b * gamma]
        rhs = 2 * f * b * (1 + reflectivity) / (1 - reflectivity)
        drhs = 4 * f * b * 1 / (1 - reflectivity) ** 2

        return lhs, rhs, drhs
