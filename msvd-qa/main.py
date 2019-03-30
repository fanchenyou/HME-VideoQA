import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dataset as dt

from attention_module_lite import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getInput(vgg, c3d, questions, answers, question_lengths, encode_answer=True):

    bsize = len(vgg)
    vgg = np.array(vgg).astype(np.float32)
    c3d = np.array(c3d).astype(np.float32)
    questions = np.array(questions).astype(np.int64)
    
    if encode_answer:
        answers = np.array(answers).astype(np.int64)

    assert vgg.shape[1]==20
    video_lengths = [vgg.shape[1]]*bsize

    vgg = torch.from_numpy(vgg).to(device)
    c3d = torch.from_numpy(c3d).to(device)        
    question_words = torch.from_numpy(questions).to(device)
    if encode_answer:
        answers = torch.from_numpy(answers).to(device,non_blocking=True)

    video_features = torch.cat([c3d,vgg],dim=2)
    video_features = video_features.view(video_features.size(0),video_features.size(1),1,1,video_features.size(2))


    data_dict = {}
    data_dict['video_features'] = video_features
    data_dict['video_lengths'] = video_lengths
    data_dict['question_words'] = question_words
    data_dict['answers'] = answers
    data_dict['question_lengths'] = question_lengths
    
    return data_dict

def main():
    """Main script."""
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--save_path', type=str, default='./saved_models/',
                        help='path for saving trained models')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    args = parser.parse_args()
    
    
    args.dataset = 'msvd_qa'
    
    args.word_dim = 300
    args.vocab_num = 4000
    args.pretrained_embedding = 'data/msvd_qa/word_embedding.npy'
    args.video_feature_dim = 4096
    args.video_feature_num = 20
    args.answer_num = 1000
    args.memory_dim = 256
    args.batch_size = 32
    args.reg_coeff = 1e-5
    args.learning_rate = 0.001
    args.preprocess_dir = 'data/msvd_qa'
    args.log = './logs'
    args.hidden_size = 512
    args.image_feature_net = 'concat'
    args.layer = 'fc'
    dataset = dt.MSVDQA(args.batch_size, args.preprocess_dir)

    args.memory_type='_mrm2s'
    args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)



    #############################
    # get video feature dimension
    #############################
    feat_channel = args.video_feature_dim
    feat_dim = 1
    text_embed_size = args.word_dim
    answer_vocab_size = args.answer_num
    voc_len = args.vocab_num
    num_layers = 2
    max_sequence_length = args.video_feature_num
    word_matrix = np.load(args.pretrained_embedding)
    answerset = pd.read_csv('data/msvd_qa/answer_set.txt', header=None)[0]


    rnn = AttentionTwoStream(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                         voc_len, num_layers, word_matrix, answer_vocab_size = answer_vocab_size,
                         max_len=max_sequence_length)
    rnn = rnn.cuda()

    if args.test == 1:
        rnn.load_state_dict(torch.load(os.path.join(args.save_model_path, 'rnn-3800-vl_0.316.pkl')))



    # loss function
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)

    
    iter = 0
    for epoch in range(0, 10000):
        dataset.reset_train()
        
        while dataset.has_train_batch:

            if args.test==0:
                vgg, c3d, questions, answers, question_lengths = dataset.get_train_batch()

                data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
                outputs, predictions = rnn(data_dict)
                targets = data_dict['answers']

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = rnn.accuracy(predictions, targets)
                print('Train iter %d, loss %.3f, acc %.2f' % (iter,loss.data,acc.item()))

            if iter % 1000==0:
                rnn.eval()
                
                # val iterate over examples
                with torch.no_grad():
                    correct = 0
                    idx = 0
                    while dataset.has_val_example:
                        if idx%100==0:
                            print 'Val iter %d/%d' % (idx,dataset.val_example_total)
                        vgg, c3d, questions, answer, question_lengths = dataset.get_val_example()
                        data_dict = getInput(vgg, c3d, questions, None, question_lengths, False)
                        outputs, predictions = rnn(data_dict)
                        prediction = predictions.item()
                        idx += 1
                        if answerset[prediction] == answer:
                            correct += 1

                    val_acc = 1.0*correct / dataset.val_example_total
                    print correct, dataset.val_example_total
                    print('Val iter %d, acc %.3f' % (iter, val_acc))
                    dataset.reset_val()


                    correct = 0
                    idx = 0
                    while dataset.has_test_example:
                        if idx%100==0:
                            print 'Test iter %d/%d' % (idx,dataset.test_example_total)
                        vgg, c3d, questions, answer, question_lengths, _ = dataset.get_test_example()
                        data_dict = getInput(vgg, c3d, questions, None, question_lengths, False)
                        outputs, predictions = rnn(data_dict)
                        prediction = predictions.item()
                        idx += 1
                        if answerset[prediction] == answer:
                            correct += 1

                    test_acc = 1.0*correct / dataset.test_example_total
                    print correct, dataset.test_example_total
                    print('Test iter %d, acc %.3f' % (iter, test_acc))
                    dataset.reset_test()

                    if args.test == 1:
                        exit()

                    torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-vl_%.3f.pkl' %(iter,val_acc)))

                rnn.train()
            iter += 1
            


if __name__ == '__main__':
    main()
