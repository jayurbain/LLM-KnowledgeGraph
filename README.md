----

### LLM, Knowledge Graphs, and Graph Machine Learning Workshop

Explore integration of Knowledge Graphs, LLMs, and Graph Machine Learning

- Use LLMs to extract entity relation from text to automatically populate a knowledge graph.  
- Use knowledge graphs or knowledge graph embeddings to ground LLM applications (retrieval augmented generation).  
- Graph machine learning to identify entity types and relations in data.

Attendees will need to do the following prior to the workshop:

[Sign up for a free graph database account at Neo4j](https://neo4j.com/cloud/platform/aura-graph-database/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=UCGenAIutm_content=AMS-Search-SEMCO-UCGenAI-None-SEM-SEM-NonABM&utm_term=neo4j%20ai&utm_adgroup=genai-llm&gad_source=1&gclid=CjwKCAiAopuvBhBCEiwAm8jaMWv4W4m41di5rIUYlb7eMLOpE7IkngU9qEllv_6C6AO7dS1JrNUVLhoCgwoQAvD_BwE)

[Sign up for an OpenAI developer key](https://openai.com/blog/openai-api)

#### Workshop 1 - Intro   
[Knowledge Graphs, Graph ML, and LLMs](https://docs.google.com/presentation/d/1yPeEIcbIK9occWx-H8hKH81Ya37kXwW6OzUDktkoiZQ/edit?usp=sharing)

[Lab: Creating a Knowledge Graph from Text](labs/constructing_knowledge_graph.ipynb)

#### Workshop 2 - Graph Embeddings   
[Knowledge Graph Embeddings]()  
[Lab: Creating a Knowledge Graph from Text]()


#### Workshop 3 - Graph Neural Networks, graph level prediction    
[Graph Neural Networks](slides/8.%20Graph%20Neural%20Networks.pdf)

#### Note: For opening notebooks in colab

_Runtime -> Change Runtime Type -> GPU_

![Colab](colab.png)

```
# !nvidia-smi

# Add this in a Google Colab cell to install the correct version of Pytorch Geometric.
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

!pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-geometric 
```

Select _Runtime -> Restart Runtime_ and start where you left off.

[Lab: Graph Convolutional Network](labs/lab3_handson_gcn.ipynb), [Solution](workshop_solutions/lab3_handson_gcn_solution.ipynb)    
[Lab: Node Classification](labs/lab3_node_class.ipynb), [Solution](workshop_solutions/lab3_handson_gcn_solution.ipynb)     
[Lab: Graph Level Prediction](labs/lab_5_graphneuralnets_esol.ipynb), [Solution](workshop_solutions/lab_5_graphneuralnets_esol_solution.ipynb)   

Workshop 4 - Deep Generative Graph Models,  illustrative notebook   
[Learning Deep Generative Models of Graphs.pdf](slides/14.%20Learning%20Deep%20Generative%20Models%20of%20Graphs.pdf)  

- [Deep Generative Graph Learning Notebook](labs/deep_graph_generative.ipynb)   
- [Model file](labs/model.pth)


### Graph Machine Learning Course

This course provides an introduction to machine learning on graphs. 

Paste the following in the first Colab cell to install pytorch geometric (if not present):

Many important real-world datasets can be represented as a graph of relationships between objects. 
Such networks are a basic tool for modeling social networks, knowledge graphs, the Web, 
and biological systems such as protein-interaction networks. Until recently, very little attention 
has been devoted to the generalization of neural network models to such structured datasets.

This course focuses on the computational, algorithmic, and modeling challenges specific to the analysis of  graphs. 
By studying the underlying graph structure and its features, students are introduced to machine learning techniques 
and data mining tools better able to reveal insights on a variety of networks.

Fundamental questions:   
Can we take advantage of graph structure to learn better representations of 
data? Given better representations, can we make better predictions?

Topics include: representation learning and Graph Neural Networks; algorithms for the World Wide Web; reasoning over Knowledge Graphs; influence maximization; disease outbreak detection, social network analysis.

Lectures are augmented with hands-on tutorials using Jupyter Notebooks. Laboratory assignments will be completed using Python and related packages: PyTorch, PyG, NumPy, Pandas, SciPy, StatsModels, SciKit-Learn, NetworkX, and MatPlotLib.

2-2-3 (class hours/week, laboratory hours/week, credits)

Prerequisites: CS-385 Algorithms, Probability and Statistics; programming maturity, and the ability to program in Python.  

ABET: Math/Science, Engineering Topics.

Outcomes:   
- Understand the basic process of applying machine learning to graph data.
- The ability to identify, load, and prepare a graph data set for a given problem.  
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to develop and apply graph neural network algorithms for node classifcation, link detection, community detection, and graph generation.   
- The ability to apply methods to real world data sets.  
- The ability to identify, articulate, and propose a research problem related to graph machine learning.

Tools: Python and related packages for data analysis, machine learning, and visualization. Jupyter Notebooks.  

Grading:  
Weekly labs and final project: 60%   
Midterm: 20%   
Final: 20%   

Office DH425:    
T 3-4pm, Th 3-4pm 

References:  

[Graph Representation Learning by William L. Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)

[http://www.cs.cornell.edu/home/kleinber/networks-book/Networks, Crowds, and Markets: Reasoning About a Highly Connected World by David Easley and Jon Kleinberg](http://www.cs.cornell.edu/home/kleinber/networks-book/)

[Network Science by Albert-László Barabási](http://networksciencebook.com/)

[Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković](https://arxiv.org/pdf/2104.13478)

[Geometric Deep Learning](https://geometricdeeplearning.com/lectures/)

[Stanford Machine Learning on Graphs](http://web.stanford.edu/class/cs224w/)

[PyG - Pytorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest)

---

### Week 1: Intro to Network Analysis and Machine Learning on Graphs

Workshop 1 - Intro   
[1. Graph Machine Learning and Motivations](slides/1.%20Graph%20Machine%20Learning%20and%20Motivations.pdf)

Workshop 2 - Basic graph representations   
[2. Graph Representations](slides/2.%20Graph%20Representations.pdf)

[Lab 0. Data Handling of Graphs](labs/Data%20Handling%20of%20Graphs.ipynb) 

[Lab 1. Graph ML Research Topics](labs/Lab%201.%20Graph%20ML%20Research%20Topics.pdf)  

[Graph Laplacian Notebook](https://colab.research.google.com/github/Taaniya/graph-analytics/blob/master/Graph_Laplacian_and_Spectral_Clustering.ipynb#scrollTo=BW6RnVt1X-0Z)

References:   
[Graph Representation Learning by William L. Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)

[TUDataset: A collection of benchmark datasets for learning with graphs](http://graphkernels.cs.tu-dortmund.de/)

[The Emerging Field of Signal Processing on Graphs](
https://arxiv.org/pdf/1211.0053.pdf)
  
### Week 2: Link Analysis and Random Walk

Workshop 3 - PageRank for recursive message passing
[3. Link Analysis](slides/3.%20Link%20Analysis.pdf)

[Lab 2: Link Analysis](labs/Lab%202.%20Link%20Analysis.pdf)  

[Lab 2: PageRank notebook](labs/PageRank.ipynb)

References:    
[The Anatomy of a Large-Scale Hypertextual Web Search Engine](http://infolab.stanford.edu/~backrub/google.html)    

[Authoritative Sources in a Hyperlinked Environment](https://www.cs.cornell.edu/home/kleinber/auth.pdf)    

### Week 3: Node Classification, Intro to Graph Neural Networks 
 
[4. Message Passing, Node Embeddings, and Representations](slides/4.%20Message%20Passing%20and%20Representations.pdf)   

[5. Node Embeddings, Random Walk, Node2vec](slides/5.%20Node%20Embeddings.pdf)
 
[Hands-on: Node2Vec](labs/DeepWalk.ipynb)  (optional)

[Lab 3: Hands-on GCN](labs/lab3_handson_gcn.ipynb)

[Lab 3: Node Classification](labs/lab3_node_class.ipynb)

### Week 4: Machine Learning Review, Graph Neural Network Intro

[6. Machine Learning Intro](slides/6.%20Machine%20Learning%20Intro.pdf)

[Hands-on Gradient Descent Notebook](labs/gradient_descent_assignment_solution.ipynb)

[Hands-on Logistic Regression with Pytorch](labs/Building%20a%20Logistic%20Regression%20Classifier%20in%20PyTorch.ipynb)

[XOR and Logistic Regression Proof](slides/XOR_and_LogisticRegression.pdf)    

[XOR and Logistic Regression Notebook](labs/XOR.ipynb)

[7. Deep Learning Intro](slides/7.%20Deep%20Learning%20Intro.pdf)

[Lab 4. Building a Neural Network in PyTorch](labs/Lab%204.%20Building%20a%20Neural%20Network%20in%20PyTorch.ipynb)

References:   

[Simplifying Graph Convolutional Networks](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)  

[SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)  

### Week 5: Deep Graph Learning, GraphSage, Applications

[8. Graph Neural Network Intro](slides/8.%20Graph%20Neural%20Network%20Intro.pdf)  
Slides 1-16.

Workshop 5 - Graph Neural Network    
[8. Graph Neural Networks](slides/8.%20Graph%20Neural%20Networks.pdf)

[Lab 5. Graph Level Prediction](labs/lab_5_graphneuralnets_esol.ipynb)

References:

[Graph Neural Networks: A review of Methods and Applications, Zhou, 2018.
Graph Representation Learning, Hamilton, 2020. ](https://arxiv.org/abs/1812.08434)

[Graph Representation Learning, Ch. 4 Graph Neural Network Model](
https://cs.mcgill.ca/~wlh/comp766/files/chapter4_draft_mar29.pdf)
 
[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

### Week 6: Graph Convolution Networks, MIDTERM  

[MIDTERM - study guide](slides/Graph%20Machine%20Learning%20Midterm%20Study%20Guide%201_11_2023.pdf) 

[Lab 6. FINAL Project Proposal](labs/Lab%206.%20Graph%20ML%20Project%20Proposal.pdf) 

<!--
[10. Graph Attention Networks]() 
-->

### Week 7: Graph Convolution Networks, Graph Relational Networks  

[9. Graph Convolution Networks](slides/9.%20Graph%20Convolution%20Networks.pdf) 

[Lab 7.1. Hands-on Loading Graphs from CSV](labs/Loading_Graphs_from_CSV.ipynb) 

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[Optional: Additional examples of Loading Graphs from CSV](labs/tabular_to_graph.ipynb)

[10. Relational Graph Networks](slides/10.%20Relational%20Graph%20Networks.pdf)
<!--
[11. Knowledge Graph Embeddings ]() 

[12. Reasoning over Knowledge Graphs]()
-->

### Week 8: Graph Attention Networks, Recommender Systems   

[11. Graph Attention Networks](slides/11.%20Graph%20Attention%20Networks.pdf)

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[12. Graph Recommender Systems](slides/12.%20Graph%20Recommender%20Systems.pdf)

<!--
[13. Frequent Subgraph Mining with GNNs]()

[15. Community Structure in Networks]()
-->

### Week 9: Generative Graph Models   

[13. Deep Generative Models for Graphs](slides/13.%20Generative%20Models%20for%20Graphs.pdf) 

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[14. Learning Deep Generative Models of Graphs.pdf](slides/14.%20Learning%20Deep%20Generative%20Models%20of%20Graphs.pdf)

Hands On:   
- [Deep Generative Graph Learning Notebook](labs/deep_graph_generative.ipynb)   
- [Model file](labs/model.pth)   
- [Annimation Gif](labs/48313438-78baf000-e5f7-11e8-931e-cd00ab34fa50.gif)

References:      
- GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec  
https://arxiv.org/abs/1802.08773
- Learning Deep Generative Models of Graphs  
https://arxiv.org/pdf/1803.03324.pdf   
- [AlphaFold](https://www.deepmind.com/research/highlighted-research/alphafold)

### Week 10: Temporal Graph Models, Final Projects, Final Exam Review 

[15. Temporal Graph Networks](slides/15.%20Temporal%20Graph%20Neural%20Networks.pdf) 

[Optional Traffice Notebook!](labs/traffic_prediction.ipynb) 

[Student presentations](labs/Lab%207.%20Graph%20ML%20Project.pdf)

[Final Exam Review](slides/Graph%20Machine%20Learning%20Final%20Exam%20Study%20Guide%202_15_2023.pdf)

### Week 11: Final Exam 

CS4981 Thursday 2:00 DH344

