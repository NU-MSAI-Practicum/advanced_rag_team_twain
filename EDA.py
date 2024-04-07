import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_TRAIN_DATA_PATH = 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
RAW_TEST_DATA_PATH = 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'



def EDA():
    
    for raw_data_path in [RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH]:
        if 'train' in raw_data_path:
            dataset = 'Train'
        else:
            dataset = 'Test'

        print('='*30)
        print(f'{dataset} data')
        
        df = pd.read_parquet(raw_data_path)

        for col in df.columns:
            print(f'  {col}')

            # Average Length
            context_lengths = df['context'].str.len()
            print(f'    Average {col} Length: {context_lengths.mean()}')

            # Max Length
            context_lengths = df['context'].str.len()
            print(f'    Max {col} Length: {context_lengths.max()}')

            # Length Distribution
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            sns.histplot(context_lengths, ax=ax)
            ax.set_title(f'{dataset} {col.capitalize()} Length Distribution')
            ax.set_xlabel(f'{dataset} {col.capitalize()} Length')
            ax.set_ylabel('Frequency')
            plt.savefig('EDAVisualizations/' + dataset + '_' + col.capitalize() + '_length_distribution.png')
            plt.close()

        print('\n\n')


if __name__ == '__main__':
    EDA()