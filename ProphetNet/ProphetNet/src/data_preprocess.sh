fairseq-preprocess --user-dir ./prophetnet --task translation_prophetnet --source-lang src --target-lang tgt --trainpref cnndm/prophetnet_tokenized/toy/train --validpref cnndm/prophetnet_tokenized/toy/valid --testpref cnndm/prophetnet_tokenized/toy/test --destdir cnndm/processed/toy --srcdict ./vocab.txt --tgtdict ./vocab.txt --workers 20