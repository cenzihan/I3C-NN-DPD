# Set up the schematic
import ads
schematic = ads.Schematic(title='Doherty PA Simulation')

# Create the components
vin = schematic.add_rfport(name='vin', port_type='input')
vdd = schematic.add_dcport(name='vdd', port_type='power', power=28)
vout = schematic.add_rfport(name='vout', port_type='output')

main_amp = schematic.add_component('MRF13750H', 'U1')
peaking_amp = schematic.add_component('MRF13750H', 'U2')

matching_network = schematic.add_component('Matching Network', 'U3')
combiner = schematic.add_component('Combiner', 'U4')

# Connect the components
schematic.connect(vin, main_amp.get_port('input'), impedance=50)
schematic.connect(vin, peaking_amp.get_port('input'), impedance=50)

schematic.connect(main_amp.get_port('output'), combiner.get_port('input1'), impedance=50)
schematic.connect(peaking_amp.get_port('output'), combiner.get_port('input2'), impedance=50)

schematic.connect(combiner.get_port('output'), matching_network.get_port('input'), impedance=50)
schematic.connect(matching_network.get_port('output'), vout, impedance=50)

# Define the simulation
sim_setup = schematic.add_simulation_setup(name='SimSetup')
sim_setup.set_frequency_range(start=900e6, stop=960e6, npts=101)

# Run the simulation
sim_results = sim_setup.simulate()

# Plot the results
import matplotlib.pyplot as plt
plt.plot(sim_results.frequency/1e6, sim_results['vin'].db, label='Input')
plt.plot(sim_results.frequency/1e6, sim_results['vout'].db, label='Output')
plt.legend()
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Doherty PA Simulation Results')
plt.show()
