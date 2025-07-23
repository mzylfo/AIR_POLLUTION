import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Dati delle stazioni Seoul (101-116)
stations_data = {
    101: {"name": "Jongno-gu", "lat": 37.572016, "lon": 127.005007},
    102: {"name": "Jung-gu", "lat": 37.564263, "lon": 126.974676},
    103: {"name": "Yongsan-gu", "lat": 37.540033, "lon": 127.004850},
    104: {"name": "Eunpyeong-gu", "lat": 37.609823, "lon": 126.934848},
    105: {"name": "Seodaemun-gu", "lat": 37.593742, "lon": 126.949679},
    106: {"name": "Mapo-gu", "lat": 37.555580, "lon": 126.905598},
    107: {"name": "Seongdong-gu", "lat": 37.541864, "lon": 127.049659},
    108: {"name": "Gwangjin-gu", "lat": 37.547180, "lon": 127.092493},
    109: {"name": "Dongdaemun-gu", "lat": 37.575743, "lon": 127.028885},
    110: {"name": "Jungnang-gu", "lat": 37.584849, "lon": 127.094023},
    111: {"name": "Seongbuk-gu", "lat": 37.606719, "lon": 127.027279},
    112: {"name": "Gangbuk-gu", "lat": 37.647930, "lon": 127.011952},
    113: {"name": "Dobong-gu", "lat": 37.654192, "lon": 127.029088},
    114: {"name": "Nowon-gu", "lat": 37.658774, "lon": 127.068505},
    115: {"name": "Yangcheon-gu", "lat": 37.525939, "lon": 126.856603},
    116: {"name": "Gangseo-gu", "lat": 37.544640, "lon": 126.835151}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcola distanza in km tra due punti geografici"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371  # Raggio Terra in km

def create_seoul_graph(distance_threshold=8.0, min_connections=2):
    """
    Crea grafo basato su vicinanza geografica
    distance_threshold: distanza massima in km per connessione
    min_connections: numero minimo di connessioni per nodo
    """
    # Mappa station code a indice (0-15)
    station_codes = list(stations_data.keys())
    code_to_index = {code: i for i, code in enumerate(station_codes)}
    
    # Calcola distanze tra tutte le coppie
    distances = {}
    for i, code1 in enumerate(station_codes):
        for j, code2 in enumerate(station_codes):
            if i != j:
                lat1, lon1 = stations_data[code1]["lat"], stations_data[code1]["lon"]
                lat2, lon2 = stations_data[code2]["lat"], stations_data[code2]["lon"]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                distances[(i, j)] = dist
    
    # Crea connessioni basate su distanza
    edges = []
    for i in range(16):
        # Trova stazioni vicine
        neighbors = []
        for j in range(16):
            if i != j and distances[(i, j)] <= distance_threshold:
                neighbors.append((j, distances[(i, j)]))
        
        # Ordina per distanza e prendi le più vicine
        neighbors.sort(key=lambda x: x[1])
        
        # Assicura numero minimo di connessioni
        num_connections = max(min_connections, len(neighbors)//2)
        num_connections = min(num_connections, len(neighbors))
        
        for neighbor, _ in neighbors[:num_connections]:
            edges.append((i, neighbor))
    
    return edges

def save_graph_to_csv(edges, filename="SEOUL_SLOT_GRAPH_n16__edges.csv"):
    """Salva grafo in formato CSV compatibile con Chengdu"""
    sources = [edge[0] for edge in edges]
    targets = [edge[1] for edge in edges]
    
    # Crea DataFrame con due righe
    df = pd.DataFrame([sources, targets])
    
    # Salva senza header e indici
    df.to_csv(filename, header=False, index=False)
    print(f"Grafo salvato in: {filename}")
    print(f"Numero di edge: {len(edges)}")
    
    return filename

def analyze_graph(edges):
    """Analizza proprietà del grafo"""
    print("\n=== ANALISI GRAFO SEOUL ===")
    print(f"Nodi: 16 (stazioni 101-116)")
    print(f"Edges: {len(edges)}")
    
    # Conta connessioni per nodo
    out_degree = {}
    in_degree = {}
    for source, target in edges:
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1
    
    print(f"Grado medio out: {np.mean(list(out_degree.values())):.2f}")
    print(f"Grado medio in: {np.mean(list(in_degree.values())):.2f}")
    
    # Mostra stazioni con più connessioni
    station_codes = list(stations_data.keys())
    print("\nStazioni più connesse (out-degree):")
    for node in sorted(out_degree.keys(), key=lambda x: out_degree[x], reverse=True)[:5]:
        station_name = stations_data[station_codes[node]]["name"]
        print(f"  {station_codes[node]} ({station_name}): {out_degree[node]} connessioni")

# ESECUZIONE
print("Creazione grafo Seoul per stazioni air pollution...")

# Genera grafo con parametri ottimizzati
edges = create_seoul_graph(distance_threshold=7.5, min_connections=3)

# Analizza
analyze_graph(edges)

# Salva
filename = save_graph_to_csv(edges)

print(f"\nFile creato: {filename}")
print("Formato compatibile con il dataset Chengdu!")