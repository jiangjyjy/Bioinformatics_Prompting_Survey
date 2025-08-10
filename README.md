# Biological Sequence with Language Model Prompting: A Survey

> A summary of the survey paper investigating how prompt-based methodologies enable Large Language Models (LLMs) to tackle challenges in biological sequence analysis. The paper covers applications across DNA, RNA, proteins, and drug discovery.

**Paper Link:** 🔗 [https://arxiv.org/abs/2503.04135](https://arxiv.org/abs/2503.04135)

---

## 📜 Abstract

Large Language Models (LLMs) are emerging as powerful tools in bioinformatics. This survey systematically investigates how prompt-based methodologies guide LLMs in analyzing biological sequences like DNA, RNA, and proteins, as well as in tasks related to drug discovery. The paper explores how prompt engineering allows LLMs to address domain-specific problems, such as promoter sequence prediction and drug-target binding affinity, often in scenarios with limited labeled data. It highlights the transformative potential of prompting while also discussing key challenges like data scarcity, multimodal fusion, and computational limitations. This work serves as a foundational resource for researchers in this rapidly evolving field.

---

## 📖 Prompting Applications in Biological Sequences

This table summarizes the key methods discussed in the survey, categorized by biological domain. It details the core prompting strategy and the specific task each method addresses.

| Domain | Method / Paper | Core Idea & Prompting Strategy | Task Addressed |
| :--- | :--- | :--- | :--- |
| 🧬 **DNA** | **NexLeth** (Zhang et al., 2024a) | Integrates knowledge graphs with personalized prompt templates (e.g., "Explain the mechanism of synthetic lethality of gene A and B"). | Generating natural language explanations for synthetic lethality (SL) mechanisms. |
| | **PLPMpro** (Li et al., 2023) | Employs prompt-learning with soft, continuous templates and verbalizers on pre-trained models like DNABERT. | Promoter sequence prediction. |
| 🦠 **RNA** | **PathoTME** (Meng et al., 2024) | Uses Visual Prompt Tuning (VPT) by adding learnable "cue vectors" to image features, combined with genomic data. | Tumor microenvironment (TME) subtype prediction. |
| | **GPTCellType** (Hou and Ji, 2024) | Leverages GPT-4 with natural language prompts that include gene lists and tissue names (e.g., "Identify cell types of [Tissue] cells by [gene list]"). | Automated cell type annotation in single-cell RNA-seq data. |
| 🔬 **Protein** | **InterDiff** (Wu et al., 2024) | A diffusion model guided by "interaction prompts" that specify desired molecular interactions (e.g., hydrogen bonds, π-π stacking). | Guided molecular generation for specific protein-ligand interactions. |
| | **Linker-Tuning** (Zou et al., 2023) | A lightweight adaptation method that inserts a learnable, continuous "Linker" prompt between protein sequences. | Optimizing heterodimeric protein structure prediction. |
| | **InstructProtein** (Wang et al., 2023) | Utilizes instruction tuning with prompts derived from knowledge graphs to bridge protein and human languages. | Zero-shot protein function annotation and de novo sequence design. |
| | **PromptMSP** (Gao et al., 2024) | Integrates conditional Protein-Protein Interaction (PPI) knowledge through prompts in a meta-learning framework. | Enhancing multimer structure prediction. |
| | **ConfProtein** (Zhang et al., 2022) | Injects both sequence and interactional conformational information into pre-trained models via specialized prompts. | Improving PPI prediction and antibody binding analysis. |
| | **ProLLM** (Jin et al., 2024) | Implements a Protein Chain of Thought (ProCoT) mechanism, translating biological signaling pathways into natural language prompts for inference. | Predicting direct and indirect protein-protein interactions. |
| 💊 **Drug Discovery** | **HGTDP-DTA** (Xiao et al., 2024) | A hybrid Graph-Transformer framework that uses dynamic prompt generation to create context-specific hints for each drug-target pair. | Drug-Target binding Affinity (DTA) prediction. |
| | **Latent Prompt Transformer** (Kong et al., 2024) | A generative model that incorporates learnable latent prompts into a unified architecture to guide molecule generation. | Multi-objective molecule optimization and drug-like molecule design. |
| | **In-Context Learning for Drug Synergy** (Edwards et al., 2023) | An in-context learning strategy that provides known drug synergy examples (selected via graph-based methods) as prompts. | Personalized prediction of synergistic drug combinations. |

---

## 🎯 Key Task Areas

The survey highlights how prompt engineering reframes complex biological problems into NLP tasks:

*   **🧬 DNA:**
    *   **Promoter Identification:** Rephrased as a masked language modeling or text classification task to find regulatory regions.
    *   **Mechanism Explanation:** Using prompts to generate human-readable explanations for complex interactions like synthetic lethality.

*   **🦠 RNA:**
    *   **Functional Element Analysis:** Identifying binding sites or regulatory elements by prompting an LLM with an RNA sequence.
    *   **Cell Type Annotation:** Classifying cells by providing gene expression profiles within a prompt to an LLM like GPT-4.

*   **🔬 Protein:**
    *   **Structure & Interaction Modeling:** Predicting 3D structures or binding modes by providing amino acid sequences or molecular pairs as context in a prompt.
    *   **De Novo Design:** Generating novel protein sequences with desired functions by providing a set of constraints and goals in an instruction-like prompt.

*   **💊 Drug Discovery:**
    *   **Binding Affinity Prediction:** Framing the problem as a question-answering task where the prompt includes the drug and target molecules.
    *   **Molecular Design:** Generating novel molecules by giving the LLM a prompt with target pharmaceutical properties (e.g., IC50, LogP).

---

## 🚧 Challenges & Future Directions

### Key Challenges
*   **Data Scarcity:** High-quality, labeled biological datasets are expensive and difficult to obtain, which limits the training and validation of models.
*   **Multimodal Fusion:** Effectively integrating diverse data types (sequence, structure, image, text) within a unified prompt framework is a significant technical hurdle.
*   **Computational Cost:** Large-scale models like AlphaFold and ESM demand substantial computational resources, limiting their accessibility for many research groups.

### Future Directions
*   **Data-Centric Annotation:** Using semi-supervised learning and generative models to create synthetic data or guide annotation, thereby overcoming data scarcity.
*   **Multi-Modal Prompting:** Developing advanced architectures that can unify structural, genomic, and other data types to capture complex biological correlations.
*   **Lightweight & Efficient Adaptation:** Employing techniques like LoRA, model pruning, and quantization to reduce the computational cost and make powerful models more accessible.

---

## ©️ Citation

If you find this survey useful in your work, please cite the original paper.

```bibtex
@article{jiang2025biological,
  title={Biological sequence with language model prompting: A survey},
  author={Jiang, Jiyue and Wang, Zikang and Shan, Yuheng and Chai, Heyan and Li, Jiayi and Ma, Zixian and Zhang, Xinrui and Li, Yu},
  journal={arXiv preprint arXiv:2503.04135},
  year={2025}
}
