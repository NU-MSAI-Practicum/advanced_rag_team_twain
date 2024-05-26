import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analysis(df1, df2, labels, title):
  sns.set_style("whitegrid")
  fig, axs = plt.subplots(1,3, figsize=(12, 5))
  for i,col in enumerate(df1.columns):
    sns.histplot(data=[df1[col].values,df2[col].values],legend=False,ax=axs[i],fill=True)
    #sns.kdeplot(data=[df1[col].values,df2[col].values],legend=False,ax=axs[i],fill=True)
    axs[i].set_title(f'{col} scores distribution')
    axs[i].legend(labels=labels)
  plt.tight_layout()
  plt.savefig(f'eda/EDAVisualizations/{title}.png')


if __name__ == '__main__':
  
    # Analysis 1 - Approx. vs. Deterministic retrieval
    # df1 = pd.read_csv('evals/eval_results_Train_100_human_labels.csv')
    # df2 = pd.read_csv('evals/es_dense_train100.csv')
    # assert df1.shape[0] == df2.shape[0]
    # analysis(
    #     #df1[['context_relevancy', 'context_precision', 'answer_correctness']],
    #     #df2[['context_relevancy', 'context_precision', 'answer_correctness']],
    #     df1,
    #     df2,
    #     labels = ['Approx. Retrieval', 'Deterministic Retrieval'],
    #     title = 'Approx. vs. Deterministic Retrieval'
    # ) 


    # # # Analysis 2 - Basic vs. Reranked Contexts
    # df1 = pd.read_csv('evals/es_dense_train100.csv')
    # df2 = pd.read_csv('evals/es_dense_train100_reranked.csv')
    # # if col == 'context_precision':
    # #     df2 = df2[df2['n_contexts'] > 1]
    # #     df1 = df1[df1.index.isin(df2.index)]

    # df2 = df2[df2['n_reranked_contexts'] > 1]
    # df1 = df1[df1.index.isin(df2.index)]
    # df1['n_reranked_contexts'] = df1['k'].copy()

    # assert df1.shape[0] == df2.shape[0]
    # analysis(
    #     df1[['context_relevancy', 'context_precision','n_reranked_contexts']],
    #     df2[['context_relevancy', 'context_precision','n_reranked_contexts']],
    #     labels = ['Basic', 'Reranked'],
    #     title = 'Basic. vs. Reranked'
    # ) 

    df1 = pd.read_csv('evals/es_dense_train100.csv')
    df2 = pd.read_csv('evals/es_dense_train100_reranked.csv')
    df_basic = df1
    df_reranking = df2
    df_basic['n_contexts'] = df_basic['k']
    df_reranking['n_contexts'] = df_reranking['n_reranked_contexts']


    df_basic['type'] = 'Basic'
    df_reranking['type'] = 'Reranking'

    # Reset the index to avoid duplicate labels
    df_basic.reset_index(drop=True, inplace=True)
    df_reranking.reset_index(drop=True, inplace=True)

    # Combine the dataframes
    #df_combined = pd.concat([df_basic, df_reranking])
    metric = 'answer_correctness'
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.histplot(data=[df_basic[metric].values, df_reranking[metric].values],  kde=True,ax=ax, bins=5)
    ax.legend(labels=['Basic','Reranking'])

    plt.show()
    pass

    # df_combined = pd.concat([df_basic, df_reranking])

    # # Plotting histograms
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # # Histogram for context_precision
    # axs[0].hist(df_basic['context_precision'], alpha=0.5, label='Basic', bins=10, color='blue')
    # axs[0].hist(df_reranking['context_precision'], alpha=0.5, label='Reranking', bins=10, color='green')
    # axs[0].set_title('Context Precision')
    # axs[0].set_xlabel('Precision')
    # axs[0].set_ylabel('Frequency')
    # axs[0].legend()

    # # Histogram for n_contexts
    # axs[1].hist(df_basic['n_contexts'], alpha=0.5, label='Basic', bins=10, color='blue')
    # axs[1].hist(df_reranking['n_contexts'], alpha=0.5, label='Reranking', bins=10, color='green')
    # axs[1].set_title('Number of Contexts')
    # axs[1].set_xlabel('Number of Contexts')
    # axs[1].set_ylabel('Frequency')
    # axs[1].legend()

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    # # Histogram for n_contexts
    # with sns.axes_style("whitegrid"):
    #     metric = 'answer_correctness'
    #     sns.histplot(data=df_combined, x=metric, hue='type', multiple='layer', kde=True, ax=axs[0])
    #     axs[0].set_title('Number of Contexts')
    #     axs[0].set_xlabel('Number of Contexts')
    #     axs[0].set_ylabel('Frequency')


    #     plt.tight_layout()
    #     plt.show()