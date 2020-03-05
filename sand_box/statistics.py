from statistics import variance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
def huberts_gamma(p_mat,c_mat):
    mat_sum = sum(p_mat[i,j] * c_mat[i,j] for i in range(len(p_mat)) for j in range(len(p_mat)) if j != i)
    m = (len(p_mat) * (len(p_mat) -1))/ 2
    return(mat_sum/m)

# normalised gamma
def norm_gamma(p_mat,c_mat):
    p_var = np.var(p_mat)
    c_var = np.var(c_mat)
    p_mean = np.mean(p_mat)
    c_mean = np.mean(c_mat)
    m = (len(p_mat) * (len(p_mat) - 1)) / 2
    mat_sum = sum((p_mat[i, j] - p_var) * (c_mat[i, j] - c_var)
                  for i in range(len(p_mat)) for j in range(len(p_mat))
                  if j != i)
    return((mat_sum/m)/(c_mean * p_mean))


# creation of c_mat
def c_mat_maker(labels):
    c_mat = [(0 if i == j else 1 ) for i in labels for j in labels]
    c_mat = np.array(c_mat)
    shape = (len(labels),len(labels))
    return(c_mat.reshape(shape))
# creation

def acc(y_true,y_pred):
    conf = confusion_matrix(y_true,y_pred)
    indx_list =
    for i in len(conf):
        indx = np.unravel_index(np.argmax(conf, axis=None), conf.shape)
        conf = np.delete(conf,indx[0], 0)
        conf = np.delete(conf, indx[1], 1)



def km_out(df,k,y):
    km = KMeans(n_clusters= k, random_state=4, n_init=40)
    km = km.fit(df)
    labels = km.labels_
    c_mat = c_mat_maker(labels)
    p_mat = pairwise_distances(df)
    match =
    out_dict = {'huberts':huberts_gamma(p_mat,c_mat),
                'norm':norm_gamma(p_mat, c_mat),
                'tss': km.initertia_,
                ''}
    return(out_dict)


km = KMeans(n_clusters=4)

km = km.fit(td2)

labels = km.labels_