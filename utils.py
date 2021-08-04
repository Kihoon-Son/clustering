import json
import networkx as nx
from numpy import genfromtxt


def read_json(all_fp_dir):
    # Use a breakpoint in the code line below to debug your script.
    with open(all_fp_dir) as json_file:
        json_data = json.load(json_file)
    print(len(json_data))
    return json_data


def find(fp_name, all_fp_json):
    for json_data in all_fp_json:
        if fp_name == json_data["name"]:
            return json_data


def read_csv(dir):
    return genfromtxt(dir, delimiter=',')


#Calculate similarities
def number_similarity(user_number_data, reference_number_data):
  count = 0
  for i in range(5):
    if user_number_data['number'][i] == reference_number_data['number'][i]:
      count += 1
  return count/5


def overallshape_similarity(user_overallshape_data, reference_overallshape_data):
  count = 0
  for i in range(64):
    if user_overallshape_data['grids'][i] == reference_overallshape_data['grids'][i]:
        count += 1
  # considering same aspect ratio
  # if((user_overallshape_data['aspect'] - reference_overallshape_data['aspect']) == 0):
  return ((1-count/64) + (abs(user_overallshape_data['aspect'] - reference_overallshape_data['aspect'])))/2


# def location_similarity(user_location_data, reference_location_data):
#   user_location_graph = nx.readwrite.node_link_graph(user_location_data)
#   referen_location_graph = nx.readwrite.node_link_graph(reference_location_data)
#   ged = gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
#   locationResult = ged.compare([user_location_graph,referen_location_graph],None)
#   return locationResult
#
# def connectivity_similarity(user_connectivity_data, reference_connectivity_data):
#   user_connectivity_graph = nx.readwrite.node_link_graph(user_connectivity_data)
#   referen_connectivity_graph = nx.readwrite.node_link_graph(reference_connectivity_data)
#   ged=gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
#   connectivityResult = ged.compare([user_connectivity_graph,referen_connectivity_graph],None)
#   return connectivityResult


# BDE 전역변수
n = 17 # 클러스터 개수
k = 0 # theta 초기화해주기 위한 임계치. n_count가 k가 되면 theta가 초기화된다.
re = 0 # re-clustering을 제안해주기 위한 임계치. is_recluster가 True가 되면 사용자에게 re-clustering을 제안한다.
prior = np.ones(n,dtype=float)/n # 최초 theta로서, 똑같이 1/n을 주며, iteration이 진행될 때마다 업데이트된다.
n_count = 0 # theta에 반하는 행동 수를 의미. k에 다다르면 다시 0으로 바뀐다.
recluster_count = 0 # recluster_count값이 re에 이르면 is_recluster를 True로 바꿔주도록 코드가 짜여있다.
previous_x = list(range(0,n)) # 최초 system에서 제공하는 view로서, iteration이 진행될 때마다 업데이트된다.
is_recluster = False # 만약에 is_recluster가 True가 되면 re-clustering을 사용자에게 제안하며, re-clustering이 진행되면 False로 초기화된다.

# BDE utils

def views(spec):
    X=list()
    for i in range(1,(len(spec)+1)):
        X+=list(itertools.combinations(spec,i))
    return(X)

def like(target,n,x):
    prob=list()
    if (type(x)==int):
        k=1
        if (not(target==x) and k==1):
            prob=[0,0.9,0.05,0.05]
        else:
            prob=[0,0.05,0.05,0.9]
    else:
        k=len(x)
        if (not(target in x) and not(k==1)):
            prob=[0.05,0.45,0.45,0.05]
        elif (target in x and k==n):
            prob=[0.95,0,0,0.05]
        else:
            prob=[(k-1)/n,(n+1-k)/(3*n),(n+1-k)/(3*n),(n+1-k)/(3*n)]
    return(prob)

def cond_ent(x,prior):
    n=len(prior)
    Theta=list(range(0,n))
    first=np.zeros(len(Theta))
    for i in range(0,len(Theta)):
        temp=like(Theta[i],n,x)
        for j in range(0,len(temp)):
            if (temp[j]==0):
                temp[j]=1
            else:
                temp[j]=temp[j]
        first[i]=sum(temp*np.log(temp))
    final=(-sum(first*prior))
    return(final)

def marg(x,prior,n_y=4):
    n=len(prior)
    Theta=list(range(0,n))
    first=np.zeros((len(Theta),n_y))
    for i in range(0,len(Theta)):
        first[i,]=like(Theta[i],n,x)
    final=np.zeros(n_y)
    for j in range(0,n_y):
        final[j]=sum(pd.DataFrame(first).loc[:,j]*prior)
    return(final)

def marg_ent(x,prior,n_y=4):
    n=len(prior)
    Theta=list(range(0,n))
    first=np.zeros(len(Theta))
    temp=marg(x,prior)
    for i in range(0,len(temp)):
        if (temp[i]==0):
            temp[i]=1
        else:
            temp[i]=temp[i]
    for j in range(0,len(Theta)):
        first[j]=sum(like(Theta[j],n,x)*np.log(temp))
    final=(-sum(first*prior))
    return(final)

def IG(x,prior,n_y=4):
    final=marg_ent(x,prior,n_y=4)-cond_ent(x,prior)
    return(final)

def max_IG(X,prior,n_y=4):
    n=len(prior)
    final=np.zeros(len(X))
    for i in range(0,len(X)):
        if i<n:
            final[i]=IG(X[i][0],prior)
        else:
            final[i]=IG(X[i],prior)
    return(final)

def posterior(x,y,prior,n_y=4):
    n=len(prior)
    Theta=list(range(0,n))
    likeli=np.zeros(len(Theta))
    for i in range(0,len(Theta)):
        likeli[i]=like(Theta[i],n,x)[y-1]
    post=(likeli*prior)/sum(likeli*prior)
    return(post)

def count_detect(user_action,clusters_in_view,prior,previous_x):
    n=len(prior)
    #Theta=list(range(0,n))
    clusters_in_view.sort()
    max_prob=max(prior) # weight of highest weight on the updated information space
    max_prob_which=np.array(range(len(prior)))[np.array(prior)==max_prob] # max probability에 해당하는 클러스터들을 인덱싱하기 위한 인덱싱 어레이라고 이해하면 된다.
    # exploration에 해당하는 행동을 진행하였을 경우에 explore에 +1을 해주기 위한 코드.
    if ((user_action==1) or (user_action==4)): # zoom-in을 했을 때 zoom-in한 cluster에 가장 높은 확률을 가지는 클러스터가 없거나, 확률이 높지 않은 클러스터를 클릭한 경우
        if ((not np.any(np.in1d(np.array(range(0,n))[max_prob_which],np.array(clusters_in_view)))) and (len(max_prob_which)<2)):
            hesitation=True
        else:
            hesitation=False
    else:
        pass
    if ((user_action==2) or (user_action==3)): # system이 제공한 이전뷰에 확률이 가장 높은 클러스터가 있었으나 zoom-out하거나 dragging한 경우
        if ((np.any(np.in1d(np.array(range(0,n))[max_prob_which],np.array(previous_x)))) and (len(max_prob_which)<2)):
            hesitation=True
        elif ((not np.any(np.in1d(np.array(range(0,n))[max_prob_which],np.array(clusters_in_view)))) and (len(max_prob_which)<2)):
            hesitation=True
        else:
            hesitation=False
    else:
        pass
    return hesitation

def Smooth_ViewSearch(user_action,clusters_in_view,prior):
    n=len(prior)
    Theta=list(range(0,n))
    if user_action==1:
        if len(clusters_in_view)==1:
            X=clusters_in_view[:]
        else:
            X=list(itertools.combinations(clusters_in_view,len(clusters_in_view)-1))
    elif user_action==2:
        if len(clusters_in_view)==n:
            X=[tuple(range(0,n))]
        else:
            if len(clusters_in_view)==n-1:
                X=[tuple(range(0,n))]
            else:
                temp=list(set(Theta)-set(clusters_in_view))
                temp.sort()
                X=np.zeros((len(temp),len(clusters_in_view)+1),dtype=int)
                for i in range(len(temp)):
                    temp2=clusters_in_view[:]
                    temp3=temp2[:]
                    temp3.append(temp[i])
                    temp3.sort()
                    X[i,]=temp3[:]
                X=X.tolist()
    elif user_action==3:
        if len(clusters_in_view)==1:
            X=clusters_in_view[:]
        else:
            X=[clusters_in_view]
    else:
        X=clusters_in_view
    # 주어진 user-input와 user-input의 specific한 양상이 주어졌을 때
    # constrained expected information gain을 최대화하는 view가 제공
    if len(X)==1:
        view=X[0]
    else:
        xsearch=max_IG(X,prior,n_y=4)
        max_index=max(xsearch)
        max_which=np.array(range(len(xsearch)))[xsearch==max_index]
        view=X[max_which[random.randint(0,len(max_which)-1)]]
        if len(view)==1:
            view=view[:][0]
    return(view)

def BIG_ViewSearch(user_action,prior):
    n=len(prior)
    Theta=list(range(0,n))
    X=views(Theta)
    xsearch=max_IG(X,prior,n_y=4) # returns a list of all the values of expected information gain for all possible views
    max_index=max(xsearch) # maximum value among all expected information gains
    max_which=np.array(range(len(xsearch)))[xsearch==max_index]
    view=X[max_which[random.randint(0,len(max_which)-1)]]
    if len(view)==1:
        view=view[:][0]
    else:
        pass
    return view