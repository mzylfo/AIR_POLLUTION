class CyclicalBetaScheduler:
    """
    Scheduler per l'annealing ciclico del coefficiente beta nel Beta-VAE
    
    Implementa una strategia di annealing ciclico dove beta varia tra un valore minimo
    e massimo secondo uno schema predefinito per ogni ciclo.
    """
    
    def __init__(
        self,
        n_epochs: int,
        n_cycles: int,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        cycle_ratio: float = 0.5,
        schedule_type: str = "ramp"
    ):
        """
        Args:
            n_epochs: Numero totale di epoche
            n_cycles: Numero di cicli di annealing
            beta_min: Valore minimo di beta
            beta_max: Valore massimo di beta
            cycle_ratio: Rapporto tra fase di ramp-up e lunghezza totale del ciclo
            schedule_type: Tipo di scheduling ('ramp' o 'cosine')
        """
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cycle_ratio = cycle_ratio
        self.schedule_type = schedule_type
        
        # Calcola la lunghezza di ogni ciclo
        self.cycle_length = n_epochs // n_cycles
        
        # Genera lo schedule completo
        self.beta_schedule = self._generate_schedule()
        
    def _generate_schedule(self) -> np.ndarray:
        """Genera lo schedule completo dei valori beta"""
        schedule = []
        
        for cycle in range(self.n_cycles):
            cycle_schedule = []
            ramp_length = int(self.cycle_length * self.cycle_ratio)
            constant_length = self.cycle_length - ramp_length
            
            if self.schedule_type == "ramp":
                # Fase di ramp-up lineare
                ramp = np.linspace(self.beta_min, self.beta_max, ramp_length)
                # Fase costante
                constant = np.ones(constant_length) * self.beta_max
                
            elif self.schedule_type == "cosine":
                # Usa una funzione coseno per una transizione più smooth
                t = np.linspace(0, np.pi, ramp_length)
                ramp = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 - np.cos(t))
                constant = np.ones(constant_length) * self.beta_max
                
            cycle_schedule.extend(ramp)
            cycle_schedule.extend(constant)
            schedule.extend(cycle_schedule)
            
        return np.array(schedule)
    
    def get_beta(self, epoch: int) -> float:
        """
        Restituisce il valore di beta per l'epoca corrente
        
        Args:
            epoch: Epoca corrente (0-based)
            
        Returns:
            float: Valore di beta per l'epoca specificata
        """
        if epoch >= self.n_epochs:
            return self.beta_max
        return float(self.beta_schedule[epoch])
    
    def plot_schedule(self):
        """Visualizza lo schedule di beta"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.plot(self.beta_schedule)
        plt.title("Beta Annealing Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Beta")
        plt.grid(True)
        raise Exception("path beta_scheduler__plot_schedule")