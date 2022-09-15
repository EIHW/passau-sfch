from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def evaluate_cont(y_true, y_pred):
    return {'auc': roc_auc_score(y_true, y_pred)}

# def evaluate_discrete(y_true, y_pred, label_names, all_labels):
#     results = {}
#     # classes = np.unique(y_true)
#     for name, metric in zip(['precision', 'recall', 'f1'], [precision_score, recall_score, f1_score]):
#         results.update({
#             f'{name}.micro': metric(y_true, y_pred, average='micro', labels=all_labels),
#             f'{name}.macro': metric(y_true, y_pred, average='macro', labels=all_labels),
#             f'{name}.weighted': metric(y_true, y_pred, average='weighted', labels=all_labels)
#         })
#
#         class_wise = metric(y_true, y_pred, average=None)
#         results.update({f'{name}.per_class.{c}': v for c,v in zip(label_names, class_wise.tolist())})
#
#     results.update({'accuracy': accuracy_score(y_true, y_pred)})
#
#     return results

# def get_confusion_matrix(y_true, y_pred, label_names):
#     labels = list(range(len(label_names)))
#     cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
#     dct = {}
#     for i,j in product(list(range(len(label_names))), list(range(len(label_names)))):
#         dct[f'{label_names[i]}.{label_names[j]}'] = float(cm[i][j])
#     return dct

# def plot_confusion_matrix_dict(conf_dict, title, savepath):
#     '''
#     For mean and std
#     '''
#     means = []
#     stds = []
#     for k1 in sorted(conf_dict.keys()):
#         mean_ls = []
#         std_ls = []
#         for k2 in sorted(conf_dict.keys()):
#             mean_ls.append(conf_dict[k1][k2]['mean'])
#             std_ls.append(conf_dict[k1][k2]['std'])
#         means.append(mean_ls)
#         stds.append(std_ls)
#     plt.clf()
#     label_names = list(sorted(conf_dict.keys()))
#     sbs.heatmap(means, annot=stds, xticklabels=label_names, yticklabels=label_names, cmap='bone_r',
#                 vmin=0, vmax=1)
#     plt.title(title)
#     plt.savefig(savepath, bbox_inches='tight')
#     pass