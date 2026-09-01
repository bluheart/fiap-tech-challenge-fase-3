import pandas as pd
from datasets import load_dataset
import os

def prepare_clinical_dataset():
    print("Carregando o dataset Texto Clinico Brasileiro...")
  
    ds = load_dataset("fabianonbfilho/texto-clinico-brasileiro", split="train")
    ds.set_format(type="pandas")

    df = ds[:]

    print(f"Dataset carregado com {len(df)} registros.")

  
    print("Criando uma coluna de classificação simulada...")

    specialties_urgent = ['Cardiologista', 'Neurologista', 'Oncologista']
    df['classificacao'] = df['specialty'].apply(
        lambda x: 'urgente' if x in specialties_urgent else 'normal'
    )

    df_final = df[['text', 'classificacao']]

    # Renomeia a coluna 'text' para 'texto' para corresponder ao esperado pela DAG
    df_final.rename(columns={'text': 'texto'}, inplace=True)

    # Cria o diretório de destino, se não existir
    os.makedirs('/opt/airflow/data', exist_ok=True)
    output_path = '/opt/airflow/data/laudos_treinamento.csv'

    # Salva como CSV
    df_final.to_csv(output_path, index=False)
    print(f"Dataset preparado e salvo em: {output_path}")
    print(f"Shape do dataset final: {df_final.shape}")

if __name__ == "__main__":
    prepare_clinical_dataset()