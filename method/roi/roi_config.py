# Based on local containment experiment (2026-08-05)
# 128^3 balances memory efficiency with ~60-67% full containment, 
# relying on the network to resolve local fracture boundaries rather than full long-bone shafts.

ROI_SHAPE = (128, 128, 128)  # (z, y, x)
CLICK_CHANNEL_VALUE = 1.0    # Value used to encode the click in the 2nd input channel