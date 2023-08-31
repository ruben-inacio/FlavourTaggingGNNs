from enum import IntEnum
import jax.numpy as jnp

NUM_JET_INPUT_PARAMETERS: int = 18  # count used as inputs in flavor tagging

class JetData(IntEnum):
    """ Store which indices of track inputs mean what. """
    TRACK_PT = 0
    TRACK_D0 = 1
    TRACK_Z0 = 2
    TRACK_PHI = 3
    TRACK_THETA = 4
    TRACK_RHO = 5
    TRACK_PT_FRACTION_LOG = 6  # log(track_pt / jet_pt)
    TRACK_DELTA_R = 7  # deltaR(track, jet)
    TRACK_PT_ERR = 8
    TRACK_D0_ERR = 9
    TRACK_Z0_ERR = 10
    TRACK_PHI_ERR = 11
    TRACK_THETA_ERR = 12
    TRACK_RHO_ERR = 13
    TRACK_SIGNED_SIG_D0 = 14  # signed d0 significance
    TRACK_SIGNED_SIG_Z0 = 15  # signed z0 significance
    # begin true production vertex info
    TRACK_PROD_VTX_X = 16
    TRACK_PROD_VTX_Y = 17
    TRACK_PROD_VTX_Z = 18
    # begin coordinates of true hadron decay for b,c jets ((0,0,0) otherwise)
    HADRON_X = 19
    HADRON_Y = 20
    HADRON_Z = 21
    # begin jet info
    N_TRACKS = 22
    TRACK_VERTEX_INDEX = 23
    # begin jet-level variables (really should be renamed to only be JET_X not TRACK_JET_X)
    TRACK_JET_PHI = 24
    TRACK_JET_THETA = 25
    TRACK_JET_PT = 26
    TRACK_JET_ETA = 27
    TRACK_JET_FLAVOR = 28
    # 29-43 binary track-level variables for training the vertex pairs auxiliary task
    # 44-46 binary jet-level variables for training the jet flavor task
    # 47-50 binary track-level variables for training the track origin auxiliary task
    TRACK_FROM_B = 47
    TRACK_FROM_C = 48
    TRACK_FROM_ORIGIN = 49
    TRACK_FROM_OTHER = 50