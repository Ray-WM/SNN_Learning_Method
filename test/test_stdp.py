import nest

if __name__ == "__main__":
    neuron1 = nest.Create("iaf_psc_alpha")
    neuron2 = nest.Create("iaf_psc_alpha")

    sp1 = nest.Create("spike_generator")
    sp2 = nest.Create("spike_generator")

    nest.Connect(neuron1, neuron2, syn_spec={'model': 'stdp_synapse'})
    nest.Connect(sp1, neuron1, syn_spec={"weight": 2000.})
    nest.Connect(sp2, neuron2, syn_spec={"weight": 2000.})

    nest.SetStatus(sp1, {'spike_times': [23.]})
    nest.SetStatus(sp2, {'spike_times': [50.]})

    # nest.SetStatus(neuron2, {'tau_minus': 0.})

    nest.Simulate(500.)
    print(nest.GetStatus(nest.GetConnections(neuron1, neuron2), ['source', 'target', 'weight']))
    cn = 0

    for i in range(100):
        if i % 2 == 0:
            # pass
            nest.SetStatus(nest.GetConnections(neuron1, neuron2), {'tau_plus': 10., 'mu_plus': 10., 'mu_minus': 10.})
            nest.SetStatus(sp2, {'spike_times': []})
        else:
            cn += 1
            nest.SetStatus(nest.GetConnections(neuron1, neuron2), {'tau_plus': 20., 'mu_plus': 1., 'mu_minus': 1.})
            nest.SetStatus(sp2, {'spike_times': [50.]})

        nest.SetStatus(sp1, {'origin': nest.GetKernelStatus()['time']})
        nest.SetStatus(sp2, {'origin': nest.GetKernelStatus()['time']})
        nest.Simulate(500.)
        print(nest.GetStatus(nest.GetConnections(neuron1, neuron2), ['source', 'target', 'weight']))

    print(cn)

    # print(nest.GetStatus(nest.GetConnections(neuron1, neuron2)))
