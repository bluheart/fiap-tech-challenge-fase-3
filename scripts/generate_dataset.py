from pathlib import Path

from datasets import load_dataset


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

    #pegando menos linhas para diminuir o tamanho do arquivo, para evitar large file no github
    df_final = df_final.sample(n=200000)

    # Renomeia a coluna 'text' para 'texto' para corresponder ao esperado pela DAG
    df_final.rename(columns={'text': 'texto'}, inplace=True)

    script_dir = Path(__file__).parent.absolute()


    output_path = script_dir.parent / 'shared/data'
    
    output_path.mkdir(parents=True, exist_ok=True)
    # Salva como CSV
    data_path = output_path / 'laudos_treinamento.parquet'

    df_final.to_parquet(data_path, index=False, compression='zstd')
    print(f"Dataset preparado e salvo em: {output_path}")
    print(f"Shape do dataset final: {df_final.shape}")

if __name__ == "__main__":
    prepare_clinical_dataset()