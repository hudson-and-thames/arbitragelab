"""
This module implements various codependence measures.
"""

from arbitragelab.codependence.correlation import (angular_distance, absolute_angular_distance, squared_angular_distance,
                                                   distance_correlation)
from arbitragelab.codependence.information import (get_mutual_info, get_optimal_number_of_bins,
                                                   variation_of_information_score)
from arbitragelab.codependence.codependence_matrix import (get_dependence_matrix, get_distance_matrix)
from arbitragelab.codependence.gnpr_distance import (spearmans_rho, gpr_distance, gnpr_distance)
from arbitragelab.codependence.optimal_transport import (optimal_transport_dependence)
