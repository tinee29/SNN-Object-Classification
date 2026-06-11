import samna
import samna.dynapse1 as dyn1 # type: ignore

def gen_clean_param_group():
    """Generate a Dynapse1ParameterGroup of one core which should 
    be able to silence the neurons.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = dyn1.Dynapse1ParameterGroup()
    # THR
    # ok
    param_group.param_map["IF_THR_N"].coarse_value = 5
    param_group.param_map["IF_THR_N"].fine_value = 80

    # refactory period
    param_group.param_map["IF_RFR_N"].coarse_value = 4
    param_group.param_map["IF_RFR_N"].fine_value = 128

    # leakage
    param_group.param_map["IF_TAU1_N"].coarse_value = 4
    param_group.param_map["IF_TAU1_N"].fine_value = 80

    param_group.param_map["IF_TAU2_N"].coarse_value = 7
    param_group.param_map["IF_TAU2_N"].fine_value = 255

    param_group.param_map["IF_DC_P"].coarse_value = 0
    param_group.param_map["IF_DC_P"].fine_value = 0

    param_group.param_map["NPDPIE_TAU_F_P"].coarse_value = 4
    param_group.param_map["NPDPIE_TAU_F_P"].fine_value = 80

    param_group.param_map["NPDPIE_THR_F_P"].coarse_value = 0
    param_group.param_map["NPDPIE_THR_F_P"].fine_value = 0

    param_group.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 0

    param_group.param_map["NPDPIE_TAU_S_P"].coarse_value = 4
    param_group.param_map["NPDPIE_TAU_S_P"].fine_value = 80

    param_group.param_map["NPDPIE_THR_S_P"].coarse_value = 0
    param_group.param_map["NPDPIE_THR_S_P"].fine_value = 0

    param_group.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 0

    param_group.param_map["IF_NMDA_N"].coarse_value = 0
    param_group.param_map["IF_NMDA_N"].fine_value = 0

    param_group.param_map["NPDPII_TAU_F_P"].coarse_value = 4
    param_group.param_map["NPDPII_TAU_F_P"].fine_value = 80

    param_group.param_map["NPDPII_THR_F_P"].coarse_value = 0
    param_group.param_map["NPDPII_THR_F_P"].fine_value = 0

    param_group.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_INH_F_N"].fine_value = 0

    param_group.param_map["NPDPII_TAU_S_P"].coarse_value = 4
    param_group.param_map["NPDPII_TAU_S_P"].fine_value = 80

    param_group.param_map["NPDPII_THR_S_P"].coarse_value = 0
    param_group.param_map["NPDPII_THR_S_P"].fine_value = 0

    param_group.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_INH_S_N"].fine_value = 0

    param_group.param_map["IF_AHTAU_N"].coarse_value = 4
    param_group.param_map["IF_AHTAU_N"].fine_value = 80

    param_group.param_map["IF_AHTHR_N"].coarse_value = 0
    param_group.param_map["IF_AHTHR_N"].fine_value = 0

    param_group.param_map["IF_AHW_P"].coarse_value = 0
    param_group.param_map["IF_AHW_P"].fine_value = 0

    param_group.param_map["IF_CASC_N"].coarse_value = 0
    param_group.param_map["IF_CASC_N"].fine_value = 0

    param_group.param_map["PULSE_PWLK_P"].coarse_value = 4
    param_group.param_map["PULSE_PWLK_P"].fine_value = 106

    param_group.param_map["R2R_P"].coarse_value = 3
    param_group.param_map["R2R_P"].fine_value = 85

    param_group.param_map["IF_BUF_P"].coarse_value = 3
    param_group.param_map["IF_BUF_P"].fine_value = 80

    return param_group

def gen_param_group():
    """Generate a Dynapse1ParameterGroup of one core with some synapse
    weights turned on for examples.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = dyn1.Dynapse1ParameterGroup()
    # THR
    # ok
    param_group.param_map["IF_THR_N"].coarse_value = 5
    param_group.param_map["IF_THR_N"].fine_value = 80

    # refactory period
    param_group.param_map["IF_RFR_N"].coarse_value = 4
    param_group.param_map["IF_RFR_N"].fine_value = 128

    # leakage
    param_group.param_map["IF_TAU1_N"].coarse_value = 4
    param_group.param_map["IF_TAU1_N"].fine_value = 80

    param_group.param_map["IF_TAU2_N"].coarse_value = 7
    param_group.param_map["IF_TAU2_N"].fine_value = 255

    param_group.param_map["IF_DC_P"].coarse_value = 0
    param_group.param_map["IF_DC_P"].fine_value = 0

    param_group.param_map["NPDPIE_TAU_F_P"].coarse_value = 4
    param_group.param_map["NPDPIE_TAU_F_P"].fine_value = 80

    param_group.param_map["NPDPIE_THR_F_P"].coarse_value = 4
    param_group.param_map["NPDPIE_THR_F_P"].fine_value = 80

    param_group.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 7
    param_group.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 80

    param_group.param_map["NPDPIE_TAU_S_P"].coarse_value = 4
    param_group.param_map["NPDPIE_TAU_S_P"].fine_value = 80

    param_group.param_map["NPDPIE_THR_S_P"].coarse_value = 4
    param_group.param_map["NPDPIE_THR_S_P"].fine_value = 80

    param_group.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 4
    param_group.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 80

    param_group.param_map["IF_NMDA_N"].coarse_value = 0
    param_group.param_map["IF_NMDA_N"].fine_value = 0

    param_group.param_map["NPDPII_TAU_F_P"].coarse_value = 4
    param_group.param_map["NPDPII_TAU_F_P"].fine_value = 80

    param_group.param_map["NPDPII_THR_F_P"].coarse_value = 4
    param_group.param_map["NPDPII_THR_F_P"].fine_value = 80

    param_group.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_INH_F_N"].fine_value = 0

    param_group.param_map["NPDPII_TAU_S_P"].coarse_value = 4
    param_group.param_map["NPDPII_TAU_S_P"].fine_value = 80

    param_group.param_map["NPDPII_THR_S_P"].coarse_value = 4
    param_group.param_map["NPDPII_THR_S_P"].fine_value = 80

    param_group.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 0
    param_group.param_map["PS_WEIGHT_INH_S_N"].fine_value = 0

    param_group.param_map["IF_AHTAU_N"].coarse_value = 4
    param_group.param_map["IF_AHTAU_N"].fine_value = 80

    param_group.param_map["IF_AHTHR_N"].coarse_value = 0
    param_group.param_map["IF_AHTHR_N"].fine_value = 0

    param_group.param_map["IF_AHW_P"].coarse_value = 0
    param_group.param_map["IF_AHW_P"].fine_value = 0

    param_group.param_map["IF_CASC_N"].coarse_value = 0
    param_group.param_map["IF_CASC_N"].fine_value = 0

    param_group.param_map["PULSE_PWLK_P"].coarse_value = 4
    param_group.param_map["PULSE_PWLK_P"].fine_value = 106

    param_group.param_map["R2R_P"].coarse_value = 3
    param_group.param_map["R2R_P"].fine_value = 85

    param_group.param_map["IF_BUF_P"].coarse_value = 3
    param_group.param_map["IF_BUF_P"].fine_value = 80

    return param_group

def gen_dc_params():
    """Generate a Dynapse1ParameterGroup based on silent neurons, and turned
    DC current on.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = gen_clean_param_group()

    param_group.param_map["IF_DC_P"].coarse_value = 2
    param_group.param_map["IF_DC_P"].fine_value = 150

    return param_group

def gen_stdp_params():
    """Generate a Dynapse1ParameterGroup for STDP example.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = gen_param_group()

    param_group.param_map["IF_TAU1_N"].coarse_value = 4
    param_group.param_map["IF_TAU1_N"].fine_value = 80

    param_group.param_map["IF_THR_N"].coarse_value = 4
    param_group.param_map["IF_THR_N"].fine_value = 80

    # NMDA, pre to post neurons
    param_group.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 7
    param_group.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 80

    # AMPA, spikegen to neurons
    param_group.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 6
    param_group.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 80

    return param_group

def set_params(model, dc=False, param_group=None):
    """Set 16 DYNAP-SE1  cores with the same Dynapse1ParameterGroup for examples.

    Args:
        model (samna.dynapse1.Dynapse1Model): Dynapse1Model
        dc (bool, optional): Turn DC on if True. Defaults to False.
        param_group (samna.dynapse1.Dynapse1ParameterGroup, optional): 
            Dynapse1ParameterGroup can be specified. Defaults to None.
    """
    if param_group is None:
        if dc:
            param_group = gen_dc_params()
        else:
            param_group = gen_param_group()
        
    for chip in range(4):
        for core in range(4):
            model.update_parameter_group(param_group, chip, core)
        
def set_stdp_params(model):
    """Set 16 DYNAP-SE1 cores with the same Dynapse1ParameterGroup for STDP example.

    Args:
        model (samna.dynapse1.Dynapse1Model): Dynapse1Model
    """
    param_group = gen_stdp_params()

    for chip in range(4):
        for core in range(4):
            model.update_parameter_group(param_group, chip, core)
