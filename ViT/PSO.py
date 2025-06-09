import random as rd
from sklearn.metrics import accuracy_score, classification_report, precision_score
from joblib import Parallel, delayed


def param_grid_generator():

    n_heads = rd.choice([even_no for even_no in range(2, 20) if even_no % 2 == 0])
    patch_size:int = rd.randint(30, 60)
    emb_size:int = rd.randint(64, 450)

    dropout:float = rd.uniform(0.1, 0.2)
    learning_rate:float = rd.uniform(1e-4, 3e-4)
    
    # error message appears `embed_dim must be divisible by num_heads` if emb_size/n_heads != 0
    if emb_size % n_heads == 0:
        pass
    else:
        while emb_size % n_heads !=0:
            emb_size+=1

    # erro message appears `einops.EinopsError: Shape mismatch, can't divide axis of length 224 in chunks of 20` if image.shape[0] % chunks != 0
    if 224 % patch_size == 0:
        pass
    else:
        while 224 % patch_size !=0:
            patch_size+=1

    param_grid:dict = {'learning_rate':learning_rate,
                       'patch_size':patch_size,
                       'emb_size':emb_size,
                       'dropout':dropout,
                       'n_heads':n_heads}
    
    return param_grid

def PSO_grid_generator():
   
    best_position_container:list = [] # Keeps track of best position g(t) visited by any particple up to itteration t

    # 1. Random initialization of particles/subsets of variants for hyperparameters (e.g., Learning Rate;	Optimizer; Loss Functios)
    up_lr:float = 0.1 # Upper range of learning rate - initially
    low_lr:float = 0.01 # Lower range of learning rate

    optimizers:list = [ # Available optimizers in torch
        "Adadelta",
        "Adafactor",
        "Adagrad",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD"
    ]

    loss_functions:list = [
        "L1Loss",
        "MSELoss",
        "CrossEntropyLoss",
        "CTCLoss",
        "NLLLoss",
        "PoissonNLLLoss",
        "GaussianNLLLoss",
        "KLDivLoss",
        "BCELoss",
        "BCEWithLogitsLoss",
        "MarginRankingLoss",
        "HingeEmbeddingLoss",
        "MultiLabelMarginLoss",
        "HuberLoss",
        "SmoothL1Loss",
        "SoftMarginLoss",
        "MultiLabelSoftMarginLoss",
        "CosineEmbeddingLoss",
        "MultiMarginLoss",
        "TripletMarginLoss",
        "TripletMarginWithDistanceLoss"
    ]


    hyperparameters_grid:dict = {
        'Learning_rate': [0.1, 0.01, 0.005, 0.0010, 0.00020], # rd.uniform(low_lr, up_lr),
        'Optimizer': optimizers, # optimizers[rd.randint(0, len(optimizers)-1)],
        'Loss Function': loss_functions # loss_functions[rd.randint(0, len(loss_functions)-1)]
    }

    def _init_Populaton():
        hyperparameters_grid:dict = {
            'Learning_rate': [0.1, 0.01, 0.005, 0.0010, 0.00020],
            'Optimizer': optimizers,
            'Loss Function': loss_functions
        }

        return hyperparameters_grid
    
    def evaluate_fitness(self,
                     X_train,
                     X_test,
                     y_train,
                     y_test,
                     hyperparameters):
        """
        Evaluate the fitness of a set of hyperparameters.

        Parameters:
            - estimator: The estimator object.
            - X_train: Training features.
            - X_test: Testing features.
            - y_train: Training labels.
            - y_test: Testing labels.
            - hyperparameters: The set of hyperparameters to evaluate.

        Returns:
            - score: The accuracy score of the estimator with the given hyperparameters.
        """
        # Unpack hyperparameters
        estimator_instance = self._create_estimator(hyperparameters)

        estimator_instance.fit(X_train, y_train)
        y_pred = estimator_instance.predict(X_test)
        accuracy_pso = accuracy_score(y_test, y_pred)
        return accuracy_pso
    
    def pso_hyperparameter_optimization(self,
                                    X_train,
                                    X_test,
                                    y_train,
                                    y_test,
                                    num_particles,
                                    num_iterations,
                                    c1 = 2.05,
                                    c2 = 2.05,
                                    num_jobs=-1,
                                    w=0.72984):
        """
        Perform hyperparameter optimization using Particle Swarm Optimization (PSO).

        Parameters:
            - estimator: The estimator object (e.g., KNeighborsClassifier, ViT).
            - data: The dataset.
            - target_column_index: Index of the target column in the dataset.
            - num_particles: Number of particles in the population.
            - num_iterations: Number of iterations for the PSO algorithm.
            - c1: Acceleration constant. Default value is c1 = 2.05
            - c2: Acceleration constant. Default value is c2 = 2.05
            - num_jobs: Number of parallel jobs for fitness evaluation.
            - inertia weight: Inertia constant. Default value is w=0.72984 according to the paper by M. Clerc and J. Kennedy

        Returns:
            - global_best_position: The best set of hyperparameters found.
            - global_best_fitness: The best accuracy found.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # 1. Initialize the population of particles
        hyperparameter_space = self._init_Populaton()
        progress_bar = tqdm(total=num_iterations, desc="PSO Progress")
        population:list = [] # Container 1: Population

        for _ in range(self.num_particles):
            hyperparameters = [np.random.choice(hyperparameter_space[param]) for param in hyperparameter_space]
            population.append(hyperparameters)


        # 2. Initialize velocity and best position
        velocity = [[0] * len(hyperparameter_space) for _ in range(num_particles)] # Container 2: Velocity, of each hyperparameter-set stays zero at initialization phase
        best_position = population.copy() # Container 3: Best position do not differ from population
        global_best_fitness = -float("inf") # Conainer 4: Best positions are all the same across individuals hence fitness stays constant
        global_best_position = [] # Container 5: Best position a given individual has ever reached




        # PSO optimization loop
        for _ in range(num_iterations):
            fitness = Parallel(n_jobs=num_jobs)(
                delayed(self.evaluate_fitness)(X_train, X_test, y_train, y_test, particle)
                for particle in population
            )



