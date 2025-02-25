# PETER (PErsonalized Transformer for Explainable Recommendation)

## Paper
- Lei Li, Yongfeng Zhang, Li Chen. [Personalized Transformer for Explainable Recommendation](https://lileipisces.github.io/files/ACL21-PETER-paper.pdf). ACL'21.

**A large pretrained GPT-2 version is available at [PEPLER](https://github.com/lileipisces/PEPLER)!**

**A small ecosystem for Recommender Systems-based Natural Language Generation is available at [NLG4RS](https://github.com/lileipisces/NLG4RS)!**

## Datasets to [download](https://drive.google.com/drive/folders/1yB-EFuApAOJ0RzTI0VfZ0pignytguU0_?usp=sharing)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

For those who are interested in how to obtain (feature, opinion, template, sentiment) quadruples, please refer to [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide).

## Usage
Below are examples of how to run PETER (with and without the key feature).
```
python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisorf/ \
--peter_mask \
--use_feature >> tripadvisorf.log

python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisor/ \
--peter_mask >> tripadvisor.log
```

## Code dependencies
- Python 3.6
- PyTorch 1.6

## Code references
- [Word Language Model](https://github.com/pytorch/examples/blob/master/word_language_model)
- [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Deploying a Seq2Seq Model with TorchScript](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)

## Citation
```
@inproceedings{ACL21-PETER,
	title={Personalized Transformer for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={ACL},
	year={2021}
}
```
