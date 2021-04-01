# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
Test functions for the Jurek OU model in the Stochastic Control Approach module.
"""
import warnings
import unittest
import os
from unittest import mock
import numpy as np
import pandas as pd

from arbitragelab.stochastic_control_approach.ou_model_jurek import StochasticControlJurek

# pylint: disable=protected-access

class TestOUModelJurek(unittest.TestCase):
    """
    Test Jurek OU model in Stochastic Control Approach module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up data and parameters.
        """

        np.random.seed(0)

        project_path = os.path.dirname(__file__)
        cls.path = project_path + '/test_data/gld_gdx_data.csv'
        data = pd.read_csv(cls.path)
        data = data.set_index('Date')

        cls.dataframe = data[['GLD', 'GDX']]


    def test_fit(self):
        """
        Tests the fit method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=True, significance_level=0.95)

        spread_value = [0.5588694237684244, 0.556388954154023, 0.5557158550545369, 0.5444317219668442,
                        0.5440914321753456, 0.5471230270552558, 0.5566709779363245, 0.5532053519529174,
                        0.5516073438832347, 0.5489831227316044, 0.5466971132268911, 0.5434512050224662,
                        0.5453702735903314, 0.5424140575831877, 0.5450224920172433, 0.5421415318431051,
                        0.5399089411892184, 0.5417342658455901, 0.5440976512434289, 0.5467950767477552,
                        0.5515669810213595, 0.5561602409477461, 0.557182448471494, 0.5561335579963324,
                        0.5566339956831441, 0.5521263613656556, 0.5390771353290404, 0.5433116196626523,
                        0.5423901587923536, 0.5310046126730086, 0.5299496560807411, 0.5299140307516241,
                        0.5290870021197289, 0.5280183485753618, 0.5386137486358292, 0.5410873897027797,
                        0.5387603530740576, 0.5352495487153268, 0.5374129802697099, 0.5448692920210395,
                        0.5400420169347888, 0.5404828705175425, 0.5359861307007192, 0.5284024338590281,
                        0.5358697412015444, 0.5371643904164909, 0.5331592612797672, 0.5377038488055406,
                        0.5372396478215287, 0.5312962591087894, 0.5211917143212158, 0.5190644803181462,
                        0.5251666364813955, 0.5239848905417022, 0.5203248327883234, 0.5243929266377997,
                        0.5191150765020052, 0.5220658221349787, 0.5176793463326727, 0.516863975659782,
                        0.5175054097681386, 0.5117401246636695, 0.5122159389305105, 0.5177991465134326,
                        0.513038871315079, 0.5098731313766349, 0.5085175023043633, 0.5040433641772756,
                        0.5039290495180081, 0.5002238140944849, 0.4947870896437365, 0.49780569727511526,
                        0.5020354188220554, 0.5023496206998971, 0.5054215932061087]

        np.testing.assert_array_almost_equal(sc_jurek.spread, spread_value, decimal=4)

        self.assertAlmostEqual(sc_jurek.mu, 0.532823, delta=1e-4)
        self.assertAlmostEqual(sc_jurek.k, 10.2728, delta=1e-4)
        self.assertAlmostEqual(sc_jurek.sigma, 0.0743999, delta=1e-4)

        project_path = os.path.dirname(__file__)
        path = project_path + '/test_data/shell-rdp-close_USD.csv'
        data = pd.read_csv(path, index_col='Date').ffill()
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
        sc_jurek.fit(data, delta_t=1 / 252, adf_test=False)


    def test_describe(self):
        """
        Tests the describe method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with self.assertRaises(Exception):
            sc_jurek.describe()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        index = ['Ticker of first stock', 'Ticker of second stock', 'Scaled Spread weights',
                 'long-term mean', 'rate of mean reversion', 'standard deviation', 'half-life']

        data = ['GLD', 'GDX', [0.779, -0.221], 0.532823, 10.2728, 0.0743999, 0.067474]

        pd.testing.assert_series_equal(pd.Series(index=index,data=data), sc_jurek.describe(), check_exact=False, atol=1e-4)


    def test_optimal_weights(self):
        """
        Tests the optimal portfolio weights method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        with self.assertRaises(Exception):
            sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=10)

        with self.assertRaises(Exception):
            sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = -1)

        weights = sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=1)

        weights_value = [-73.45194149467186, -66.92279595239394, -65.15408289280388, -35.43777877275913,
                         -34.545817687919545, -42.535359317812855, -67.68919459098552, -58.56650579922123,
                         -54.36327112091763, -47.45713298534966, -41.44201490315255, -32.89844175560287,
                         -37.96096743214949, -30.18088183878759, -37.060772390128506, -29.479402457404348,
                         -23.60620567684073, -28.42505229728819, -34.66288294932173, -41.78249330166799,
                         -54.37101206797773, -66.49171390891671, -69.2030397247047, -66.45661770001615,
                         -67.79578295793033, -55.93171185761935, -21.540238878824916, -32.726868430709885,
                         -30.316954989694864, -0.2964470572123097, 2.4712176283324236, 2.548417241307483,
                         4.714693033871299, 7.519785928719447, -20.497495188429824, -27.06729132591058,
                         -20.947832834193054, -11.69200027231988, -17.452717588039015, -37.247605794787766,
                         -24.5030045585155, -25.72274610257665, -13.832747281667135, 6.28649720491654,
                         -13.620032093482212, -17.124887100964134, -6.494784173912921, -18.6978042520508,
                         -17.529819813294633, -1.6519685678067688, 25.4931602870002, 31.25446478754843,
                         14.784765120158685, 17.99308095126069, 27.98439238126017, 16.914096722374268,
                         31.452327361923672, 23.40469539338531, 35.68396003464806, 38.1450485471757,
                         36.56085532095218, 53.20482218153802, 52.27927773305257, 36.570939606379525,
                         50.92261233207725, 60.964273675008954, 65.91794540384299, 80.72165214689629,
                         82.56422216475477, 96.13482957606092, 116.22091468441988, 109.1943316005652,
                         97.8535878146283, 100.10622031248108, 92.57663162301588]

        np.testing.assert_array_almost_equal(weights, weights_value, decimal=4)

        sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 1, utility_type=1)
        sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=2)
        sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 2, utility_type=1)
        sc_jurek.optimal_portfolio_weights(self.dataframe, beta = 0.01, gamma = 2, utility_type=2)


    def test_stabilization_region(self):
        """
        Tests the stabilization region method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        with self.assertRaises(Exception):
            sc_jurek.stabilization_region_calc(self.dataframe, beta=0.01, gamma=0.5, utility_type=10)

        with self.assertRaises(Exception):
            sc_jurek.stabilization_region_calc(self.dataframe, beta = 0.01, gamma = -1)

        spread, min_bound, max_bound = sc_jurek.stabilization_region_calc(self.dataframe, beta = 0.01, gamma = 0.5, utility_type=1)

        spread_value = [0.5588694237684244, 0.556388954154023, 0.5557158550545369, 0.5444317219668442,
                   0.5440914321753456, 0.5471230270552558, 0.5566709779363245, 0.5532053519529174,
                   0.5516073438832347, 0.5489831227316044, 0.5466971132268911, 0.5434512050224662,
                   0.5453702735903314, 0.5424140575831877, 0.5450224920172433, 0.5421415318431051,
                   0.5399089411892184, 0.5417342658455901, 0.5440976512434289, 0.5467950767477552,
                   0.5515669810213595, 0.5561602409477461, 0.557182448471494, 0.5561335579963324,
                   0.5566339956831441, 0.5521263613656556, 0.5390771353290404, 0.5433116196626523,
                   0.5423901587923536, 0.5310046126730086, 0.5299496560807411, 0.5299140307516241,
                   0.5290870021197289, 0.5280183485753618, 0.5386137486358292, 0.5410873897027797,
                   0.5387603530740576, 0.5352495487153268, 0.5374129802697099, 0.5448692920210395,
                   0.5400420169347888, 0.5404828705175425, 0.5359861307007192, 0.5284024338590281,
                   0.5358697412015444, 0.5371643904164909, 0.5331592612797672, 0.5377038488055406,
                   0.5372396478215287, 0.5312962591087894, 0.5211917143212158, 0.5190644803181462,
                   0.5251666364813955, 0.5239848905417022, 0.5203248327883234, 0.5243929266377997,
                   0.5191150765020052, 0.5220658221349787, 0.5176793463326727, 0.516863975659782,
                   0.5175054097681386, 0.5117401246636695, 0.5122159389305105, 0.5177991465134326,
                   0.513038871315079, 0.5098731313766349, 0.5085175023043633, 0.5040433641772756,
                   0.5039290495180081, 0.5002238140944849, 0.4947870896437365, 0.49780569727511526,
                   0.5020354188220554, 0.5023496206998971, 0.5054215932061087]

        min_bound_value = [0.5114954740261215, 0.511494304659523, 0.5114930719447892, 0.511491772862663,
                           0.5114904043053522, 0.5114889630823879, 0.5114874459281928, 0.5114858495116325,
                           0.5114841704478517, 0.5114824053127447, 0.5114805506604544, 0.5114786030443454,
                           0.5114765590419567, 0.5114744152845064, 0.5114721684915997, 0.5114698155118698,
                           0.5114673533703791, 0.5114647793237237, 0.5114620909238913, 0.5114592860920708,
                           0.5114563632037615, 0.5114533211866995, 0.5114501596333154, 0.511446878929659,
                           0.5114434804029621, 0.5114399664902919, 0.5114363409310583, 0.5114326089864681,
                           0.5114287776894266, 0.5114248561288008, 0.5114208557724526, 0.5114167908339828,
                           0.511412678688732, 0.5114085403452506, 0.5114044009791965, 0.5114002905374362,
                           0.5113962444210516, 0.5113923042569426, 0.5113885187688384, 0.511384944759741,
                           0.5113816482191418, 0.511378705569804, 0.511376205070451, 0.511374248392369,
                           0.5113729523897055, 0.5113724510851133, 0.5113728890138528, 0.5113744592285356,
                           0.5113773528151565, 0.511381797500334, 0.5113880522315185, 0.5113964110175347,
                           0.5114072071883844, 0.5114208181088091, 0.5114376703799909, 0.5114582455627015,
                           0.5114830864528888, 0.5115128039368494, 0.5115480844474176, 0.5115896980347043,
                           0.5116385070545585, 0.5116954754648763, 0.5117616787042263, 0.5118383141092488,
                           0.5119267118078763, 0.512028346006258, 0.512144846571341, 0.5122780108031038,
                           0.5124298152978717, 0.5126024278381169, 0.5127982193212931, 0.5130197758849269,
                           0.5132699116333537, 0.5135516827763962, 0.5138684046323441]

        max_bound_value = [0.5504663932787668, 0.5504650623703748, 0.5504636483725223, 0.5504621458535578,
                           0.550460548998529, 0.5504588515790302, 0.5504570469203628, 0.5504551278657476,
                           0.5504530867372842, 0.5504509152933343, 0.5504486046819679, 0.5504461453900639,
                           0.5504435271876233, 0.5504407390667989, 0.5504377691750882, 0.5504346047420877,
                           0.5504312319991208, 0.5504276360910032, 0.5504238009790952, 0.5504197093347314,
                           0.5504153424219909, 0.5504106799686727, 0.5504057000242086, 0.5504003788031157,
                           0.5503946905124223, 0.5503886071613505, 0.5503820983513332, 0.550375131044248,
                           0.5503676693065147, 0.5503596740264586, 0.5503511026020539, 0.5503419085958711,
                           0.5503320413537078, 0.5503214455830255, 0.5503100608869134, 0.550297821248873,
                           0.5502846544632468, 0.5502704815056058, 0.5502552158368782, 0.5502387626344095,
                           0.5502210179425316, 0.5502018677345667, 0.5501811868775095, 0.5501588379899172,
                           0.5501346701828164, 0.5501085176726986, 0.550080198254951, 0.5500495116253599,
                           0.5500162375366732, 0.5499801337766138, 0.5499409339532445, 0.5498983539501722,
                           0.5498520537857988, 0.5498016879649662, 0.5497468668698894, 0.5496871622327966,
                           0.5496221034636182, 0.5495511736884243, 0.5494738054880556, 0.5493893763277539,
                           0.5492972036697708, 0.5491965397614428, 0.5490865660904667, 0.5489663874959719,
                           0.5488350259169966, 0.5486914137468271, 0.5485343867390974, 0.5483626763748868,
                           0.5481749015425326, 0.5479695592938455, 0.5477450143079432, 0.547499486496596,
                           0.5472310358926771, 0.5469375435302057, 0.5466166863794518]


        np.testing.assert_array_almost_equal(spread, spread_value, decimal=4)
        np.testing.assert_array_almost_equal(min_bound, min_bound_value, decimal=4)
        np.testing.assert_array_almost_equal(max_bound, max_bound_value, decimal=4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.stabilization_region_calc(self.dataframe, gamma=1, utility_type=1)

        sc_jurek.stabilization_region_calc(self.dataframe, beta=0.01, gamma=0.5, utility_type=1)
        sc_jurek.stabilization_region_calc(self.dataframe, beta=0.01, gamma=2, utility_type=2)


    def test_optimal_weights_fund_flows(self):
        """
        Tests the optimal weights with fund flows method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        weights = sc_jurek.optimal_portfolio_weights_fund_flows(self.dataframe, f=0.2, gamma = 0.5)

        weights_value = [-61.209951245559886, -55.768996626994955, -54.29506907733657, -29.53148231063261,
                         -28.788181406599623, -35.446132764844045, -56.407662159154604, -48.80542149935103,
                         -45.30272593409803, -39.54761082112472, -34.53501241929379, -27.41536812966906,
                         -31.63413952679124, -25.15073486565633, -30.883976991773757, -24.566168714503625,
                         -19.671838064033942, -23.68754358107349, -28.88573579110144, -34.81874441805666,
                         -45.30917672331478, -55.40976159076393, -57.66919977058725, -55.38051475001346,
                         -56.49648579827527, -46.60975988134946, -17.950199065687432, -27.272390358924905,
                         -25.264129158079054, -0.2470392143435914, 2.059348023610353, 2.1236810344229027,
                         3.928910861559416, 6.266488273932873, -17.081245990358187, -22.556076104925484,
                         -17.456527361827547, -9.743333560266567, -14.543931323365847, -31.039671495656474,
                         -20.419170465429584, -21.43562175214721, -11.52728940138928, 5.238747670763784,
                         -11.350026744568511, -14.270739250803446, -5.412320144927434, -15.581503543375668,
                         -14.608183177745529, -1.3766404731723074, 21.244300239166837, 26.045387322957026,
                         12.320637600132239, 14.994234126050577, 23.320326984383478, 14.095080601978557,
                         26.21027280160306, 19.503912827821093, 29.736633362206717, 31.787540455979748,
                         30.467379434126816, 44.33735181794835, 43.56606477754381, 30.475783005316273,
                         42.435510276731044, 50.803561395840795, 54.93162116986916, 67.2680434557469,
                         68.80351847062897, 80.11235798005077, 96.85076223701657, 90.99527633380434,
                         81.54465651219026, 83.4218502604009, 77.14719301917991]


        np.testing.assert_array_almost_equal(weights, weights_value, decimal=4)

    def test_private_methods(self):
        """
        Function tests special cases for code coverage.
        """

        sc_jurek = StochasticControlJurek()


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc_jurek.fit(self.dataframe, delta_t=1/252, adf_test=False)

        sc_jurek.optimal_portfolio_weights(self.dataframe, beta=0.01, gamma=0.5, utility_type=1)

        time_array = np.arange(0, len(self.dataframe)) * (1/252)
        tau = time_array[-1] - time_array
        c_1 = 0.0221413
        c_2 = -20.59561
        c_3 = 9625.4458424
        disc = -844.234913

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_jurek._A_calc_1(tau, c_1, c_2, disc, 0.5)
            sc_jurek._A_calc_1(tau, c_1, c_2, disc, 0.9)
            sc_jurek._B_calc_1(tau, c_1, c_2, c_3, disc, 0.5)
            sc_jurek._B_calc_1(tau, c_1, c_2, c_3, disc, 0.9)

        sc_jurek.optimal_portfolio_weights(self.dataframe, beta=0.01, gamma=0.5, utility_type=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_jurek._B_calc_2(tau, c_1, c_2, c_3, disc, 0.5)
            sc_jurek._B_calc_2(tau, c_1, c_2, c_3, disc, 0.9)



    @mock.patch("arbitragelab.stochastic_control_approach.ou_model_jurek.plt")
    def test_plotting(self, mock_plt):
        """
        Tests the plotting method in the class.
        """

        sc_jurek = StochasticControlJurek()

        with self.assertRaises(Exception):
            sc_jurek.plotting(self.dataframe)

        self.dataframe.index = pd.to_datetime(self.dataframe.index)

        with self.assertRaises(Exception):
            sc_jurek.plotting(self.dataframe)

        project_path = os.path.dirname(__file__)
        path = project_path + '/test_data/shell-rdp-close_USD.csv'
        data = pd.read_csv(path, index_col='Date').ffill()

        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")

        sc_jurek.plotting(data)

        # Assert plt.figure got called
        assert mock_plt.show.called
