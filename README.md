# Semantic Encoding with Large Language Model for Drug Repositioning

# Abstract

Drug repositioning has been widely adopted for identifying new therapeutic indications of existing drugs. However, effective modeling of the semantics underlying drug-disease associations remains challenging, as existing methods suffer from semantic misalignment between continuous attributes and discrete structures, and excessive reliance on external biological knowledge that may introduce noise. To address these issues, ${\rm SEDR}_{LLM}$ is proposed as a structure-aware semantic enhancement framework for drug repositioning. Rather than serving as an external knowledge source, the large language model is guided by Structure-Constrained Graph Semantic Prompts to perform semantic abstraction exclusively from structural statistical observations of the bipartite graph, yielding structure-consistent and noise-controllable semantic representations. By jointly modeling bidirectionally coupled bipartite graph structures and a Semantic-Aware Siamese Encoder, high-order semantics embedded in drug-disease associations are systematically captured without relying on additional biological features. Extensive experiments on multiple public datasets demonstrate that ${\rm SEDR}_{LLM}$ outperforms state-of-the-art methods across diverse evaluation settings. A breast cancer case study further confirms its feasibility and practical value in real-world drug repositioning tasks.

# 1. Requirements

To reproduce **SEDR**, the python==3.8,pytorch==1.8.0, rdkit-pypi==2022.3.2 are required.

Of course, you can create your environment by env.yaml:
```sh
    $ conda env create -f env.yaml
```

# 2. Usage

### 2.1. Data

Data for SEDR can be downloaded from [DRGCL](https://ieeexplore.ieee.org/abstract/document/10458294/).

### 2.2. Useage 
For training:
```sh
    $ python train.py
```

### 2.3. Baselines

DRHGCN: [https://academic.oup.com/bib/article-abstract/22/6/bbab319/6347207](https://academic.oup.com/bib/article-abstract/22/6/bbab319/6347207)

DRWBNCF: [https://academic.oup.com/bib/article-abstract/23/2/bbab581/6510159](https://academic.oup.com/bib/article-abstract/23/2/bbab581/6510159)

DRGBCN: [https://ieeexplore.ieee.org/abstract/document/10324328/](https://ieeexplore.ieee.org/abstract/document/10324328/)

AMDGT: [https://www.sciencedirect.com/science/article/pii/S0950705123010778](https://www.sciencedirect.com/science/article/pii/S0950705123010778)

AdaDR: [https://academic.oup.com/bioinformatics/article-abstract/40/1/btad748/7467059](https://academic.oup.com/bioinformatics/article-abstract/40/1/btad748/7467059)

DRGCL: [https://ieeexplore.ieee.org/abstract/document/10458294/](https://ieeexplore.ieee.org/abstract/document/10458294/)

AutoDR: [https://www.sciencedirect.com/science/article/pii/S0952197624018116](https://www.sciencedirect.com/science/article/pii/S0952197624018116)

DRDM: [https://www.sciencedirect.com/science/article/pii/S1532046425000723](https://www.sciencedirect.com/science/article/pii/S1532046425000723)

HGCL-DR: [https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c00435](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c00435)

MRDDA: [https://link.springer.com/article/10.1186/s12967-025-06783-x](https://link.springer.com/article/10.1186/s12967-025-06783-x)
# 3. Concat
Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at niudongjiang@qdu.edu.cn.
