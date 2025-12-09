import os
import nest
from pynestml.frontend.pynestml_frontend import generate_nest_target


def compile_nestml_models():
    # Path to the NESTML model files
    neuron_model_file = os.path.join("/home/miguel/my_codes/neuro-quake/src/nestml/neuron_models",
                                     "aeif_cond_alpha_nq.nestml")
    synapse_model_file = os.path.join("/home/miguel/my_codes/neuro-quake/src/nestml/synapse_models",
                                      "stdp_reversal_pot_nq.nestml")

    input_path = [os.path.realpath(neuron_model_file), os.path.realpath(synapse_model_file)]

    # Generate the NEST target code and module
    generate_nest_target(input_path=input_path,
                         target_path="compiled_models",
                         module_name="stdp_reversal_pot_nq_module",
                         codegen_opts={"neuron_synapse_pairs": [{"neuron": "aeif_cond_alpha_nq",
                                                                 "synapse": "stdp_reversal_pot_nq",
                                                                 "post_ports": ["post_spikes"]}]})

    # Install the module in NEST
    nest.Install("stdp_reversal_pot_nq_module")


# Call the function to compile and install the NESTML module
compile_nestml_models()
