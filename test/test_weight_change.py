import nest

if __name__ == "__main__":

    pop1 = nest.Create(5)
    pop2 = nest.Create(3)

    nest.Connect(pop1, pop2, conn_spec="all_to_all")
