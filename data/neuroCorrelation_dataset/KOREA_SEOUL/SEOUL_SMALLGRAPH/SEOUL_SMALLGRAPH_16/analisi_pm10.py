import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')

class CSVOutlierAnalyzer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        self.all_values = []
        self.stats_dict = {}
        self.outlier_bounds = {}
        
    def load_data(self):
        """Carica i dati dal CSV"""
        print("Caricamento dati...")
        self.df = pd.read_csv(self.input_file)
        print(f"Caricate {len(self.df)} righe")
        
        # Assumo che la prima colonna sia il riferimento e la seconda l'array
        # Modifica questi nomi se necessario
        if len(self.df.columns) == 2:
            self.df.columns = ['riferimento', 'valori_array']
        
        return self.df
    
    def parse_arrays(self):
        """Converte le stringhe degli array in liste di numeri"""
        print("Parsing degli array...")
        parsed_arrays = []
        
        for idx, row in self.df.iterrows():
            try:
                # Prova diversi metodi per parsare l'array
                array_str = str(row['valori_array'])
                
                # Rimuovi spazi e caratteri indesiderati
                array_str = array_str.strip('[]').replace('\n', '').replace('\r', '')
                
                # Prova con ast.literal_eval
                try:
                    if array_str.startswith('[') and array_str.endswith(']'):
                        values = ast.literal_eval(array_str)
                    else:
                        # Split per virgola e converti
                        values = [float(x.strip()) for x in array_str.split(',') if x.strip()]
                except:
                    # Fallback: split per spazi o virgole
                    values = []
                    for item in array_str.replace(',', ' ').split():
                        try:
                            values.append(float(item))
                        except:
                            continue
                
                if len(values) != 2700:
                    print(f"Attenzione: riga {idx} ha {len(values)} valori invece di 2700")
                
                parsed_arrays.append(values)
                self.all_values.extend(values)
                
            except Exception as e:
                print(f"Errore nel parsing della riga {idx}: {e}")
                parsed_arrays.append([])
        
        self.df['parsed_values'] = parsed_arrays
        print(f"Totale valori analizzati: {len(self.all_values)}")
        
    def analyze_distribution(self):
        """Analizza la distribuzione di tutti i valori"""
        print("\nAnalisi della distribuzione...")
        
        if not self.all_values:
            raise ValueError("Nessun valore trovato per l'analisi")
        
        values_array = np.array(self.all_values)
        
        # Statistiche descrittive
        self.stats_dict = {
            'count': len(values_array),
            'mean': np.mean(values_array),
            'median': np.median(values_array),
            'std': np.std(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'q1': np.percentile(values_array, 25),
            'q3': np.percentile(values_array, 75)
        }
        
        # Calcola IQR per identificare outlier
        iqr = self.stats_dict['q3'] - self.stats_dict['q1']
        self.outlier_bounds = {
            'lower': self.stats_dict['q1'] - 1.5 * iqr,
            'upper': self.stats_dict['q3'] + 1.5 * iqr
        }
        
        # Conta outlier
        outliers = values_array[(values_array < self.outlier_bounds['lower']) | 
                               (values_array > self.outlier_bounds['upper'])]
        self.stats_dict['outliers_count'] = len(outliers)
        self.stats_dict['outliers_percentage'] = (len(outliers) / len(values_array)) * 100
        
        return self.stats_dict
    
    def print_distribution_table(self):
        """Stampa tabella delle distribuzioni"""
        print("\n" + "="*50)
        print("TABELLA DISTRIBUZIONE VALORI")
        print("="*50)
        
        print(f"Numero totale valori:    {self.stats_dict['count']:,}")
        print(f"Media:                   {self.stats_dict['mean']:.4f}")
        print(f"Mediana:                 {self.stats_dict['median']:.4f}")
        print(f"Deviazione standard:     {self.stats_dict['std']:.4f}")
        print(f"Minimo:                  {self.stats_dict['min']:.4f}")
        print(f"Massimo:                 {self.stats_dict['max']:.4f}")
        print(f"Q1 (25°):               {self.stats_dict['q1']:.4f}")
        print(f"Q3 (75°):               {self.stats_dict['q3']:.4f}")
        print(f"Range interquartile:     {self.stats_dict['q3'] - self.stats_dict['q1']:.4f}")
        print(f"\nSOGLIE OUTLIER:")
        print(f"Limite inferiore:        {self.outlier_bounds['lower']:.4f}")
        print(f"Limite superiore:        {self.outlier_bounds['upper']:.4f}")
        print(f"Outlier trovati:         {self.stats_dict['outliers_count']:,}")
        print(f"Percentuale outlier:     {self.stats_dict['outliers_percentage']:.2f}%")
        print("="*50)
    
    def replace_outliers(self):
        """Sostituisce gli outlier con valori nella distribuzione normale"""
        print("\nSostituzione outlier...")
        
        replaced_count = 0
        
        # Calcola statistiche della distribuzione pulita una sola volta
        clean_values = np.array(self.all_values)
        clean_values = clean_values[(clean_values >= self.outlier_bounds['lower']) & 
                                  (clean_values <= self.outlier_bounds['upper'])]
        clean_mean = np.mean(clean_values)
        clean_std = np.std(clean_values)
        
        # Crea una nuova lista per i valori modificati
        new_parsed_values = []
        
        for idx in range(len(self.df)):
            values = self.df.iloc[idx]['parsed_values'].copy()
            
            for i in range(len(values)):
                if (values[i] < self.outlier_bounds['lower'] or 
                    values[i] > self.outlier_bounds['upper']):
                    
                    # Genera nuovo valore dalla distribuzione normale
                    new_value = np.random.normal(clean_mean, clean_std)
                    
                    # Assicurati che sia nei bounds
                    new_value = np.clip(new_value, 
                                      self.outlier_bounds['lower'], 
                                      self.outlier_bounds['upper'])
                    
                    values[i] = new_value
                    replaced_count += 1
            
            new_parsed_values.append(values)
        
        # Sostituisci tutta la colonna in una volta
        self.df['parsed_values'] = new_parsed_values
        
        print(f"Sostituiti {replaced_count} outlier")
    
    def save_results(self, output_file):
        """Salva i risultati nel nuovo CSV"""
        print(f"\nSalvataggio in {output_file}...")
        
        # Prepara il dataframe di output
        output_df = pd.DataFrame()
        output_df['riferimento'] = self.df['riferimento']
        
        # Converte gli array di nuovo in stringhe (formato originale)
        array_strings = []
        for values in self.df['parsed_values']:
            # Converte in stringa mantenendo la struttura originale
            array_str = '[' + ','.join([str(v) for v in values]) + ']'
            array_strings.append(array_str)
        
        output_df['valori_array'] = array_strings
        
        # Salva
        output_df.to_csv(output_file, index=False)
        print(f"File salvato con {len(output_df)} righe")
    
    def create_visualization(self):
        """Crea grafici della distribuzione"""
        print("\nCreazione visualizzazioni...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        values_array = np.array(self.all_values)
        
        # Istogramma
        axes[0,0].hist(values_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.stats_dict['mean'], color='red', linestyle='--', label=f'Media: {self.stats_dict["mean"]:.2f}')
        axes[0,0].axvline(self.stats_dict['median'], color='green', linestyle='--', label=f'Mediana: {self.stats_dict["median"]:.2f}')
        axes[0,0].set_title('Distribuzione dei Valori')
        axes[0,0].set_xlabel('Valore')
        axes[0,0].set_ylabel('Frequenza')
        axes[0,0].legend()
        
        # Box plot
        axes[0,1].boxplot(values_array)
        axes[0,1].set_title('Box Plot')
        axes[0,1].set_ylabel('Valore')
        
        # Q-Q plot
        stats.probplot(values_array, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot (Normalità)')
        
        # Density plot
        axes[1,1].hist(values_array, bins=50, density=True, alpha=0.7, color='lightcoral')
        
        # Aggiungi curva normale teorica
        x = np.linspace(values_array.min(), values_array.max(), 100)
        normal_curve = stats.norm.pdf(x, self.stats_dict['mean'], self.stats_dict['std'])
        axes[1,1].plot(x, normal_curve, 'r-', linewidth=2, label='Normale teorica')
        axes[1,1].set_title('Densità vs Distribuzione Normale')
        axes[1,1].set_xlabel('Valore')
        axes[1,1].set_ylabel('Densità')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('distribuzione_analisi.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configurazione
    input_file = 'data.csv'  # Modifica con il nome del tuo file
    output_file = 'output_data_clean.csv'
    
    try:
        # Inizializza analyzer
        analyzer = CSVOutlierAnalyzer(input_file)
        
        # Esegui analisi completa
        analyzer.load_data()
        analyzer.parse_arrays()
        analyzer.analyze_distribution()
        analyzer.print_distribution_table()
        analyzer.replace_outliers()
        analyzer.save_results(output_file)
        analyzer.create_visualization()
        
        print("\n✅ Analisi completata con successo!")
        print(f"📊 File pulito salvato come: {output_file}")
        print(f"📈 Grafici salvati come: distribuzione_analisi.png")
        
    except FileNotFoundError:
        print(f"❌ Errore: File {input_file} non trovato")
        print("Assicurati che il file esista nella directory corrente")
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")

if __name__ == "__main__":
    main()