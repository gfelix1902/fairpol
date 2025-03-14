import pandas as pd
import pyreadr
import os

def load_and_process_data(file_path, output_file=None):
    try:
        if os.path.exists(file_path):
            r_data = pyreadr.read_r(file_path)
            # Ausgabe der Schlüssel um den richtigen Schlüssel zu finden.
            print(r_data.keys())
            # Hier muss der richtige Schlüssel eingesetzt werden.
            data = r_data['JC']
            print(f"Daten erfolgreich aus {file_path} geladen.")
            print(f"Datenform: {data.shape}")
            if output_file:
                data.to_csv(output_file, index=False)
                print(f"Daten erfolgreich in {output_file} geschrieben.")
            print("\nErste 5 Zeilen der Daten:")
            print(data.head())
        else:
            print(f"Fehler: Datei {file_path} nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim Laden oder Verarbeiten der Daten: {e}")

def main():
    file_path = 'C:/Users/Felix/OneDrive/Dokumente/Bachelorarbeit/fairpol/data/JC.RData'
    output_file = 'JC_processed.csv'
    load_and_process_data(file_path, output_file)