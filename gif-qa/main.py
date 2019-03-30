import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from attention_module_lite import *
from make_tgif import DatasetTGIF
from util import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    torch.manual_seed(1)
    
    ### add arguments ###
    args.vc_dir = './data/Vocabulary'
    args.df_dir = './data/dataset'
    args.max_sequence_length = 35
    args.model_name = args.task + args.feat_type

    args.memory_type = '_mrm2s'
    args.image_feature_net = 'concat'
    args.layer = 'fc'


    
    args.save_model_path = args.save_path + '%s_%s_%s%s' % (args.task,args.image_feature_net,args.layer,args.memory_type)
    # Create model directory
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    

    ######################################################################################
    ## This part of dataset code is adopted from
    ## https://github.com/YunseokJANG/tgif-qa/blob/master/code/gifqa/data_util/tgif.py
    ######################################################################################
    print 'Start loading TGIF dataset'
    train_dataset = DatasetTGIF(dataset_name='train',
                                             image_feature_net=args.image_feature_net,
                                             layer=args.layer,
                                             max_length=args.max_sequence_length,
                                             data_type=args.task,
                                             dataframe_dir=args.df_dir,
                                             vocab_dir=args.vc_dir)
    train_dataset.load_word_vocabulary()

    val_dataset = train_dataset.split_dataset(ratio=0.1)
    val_dataset.share_word_vocabulary_from(train_dataset)

    test_dataset = DatasetTGIF(dataset_name='test',
                                image_feature_net=args.image_feature_net,
                                layer=args.layer,
                                max_length=args.max_sequence_length,
                                data_type=args.task,
                                dataframe_dir=args.df_dir,
                                vocab_dir=args.vc_dir)

    test_dataset.share_word_vocabulary_from(train_dataset)
    
    print 'dataset lengths train/val/test %d/%d/%d' % (len(train_dataset),len(val_dataset),len(test_dataset))
    
    
    #############################
    # get video feature dimension
    #############################
    video_feature_dimension = train_dataset.get_video_feature_dimension()
    feat_channel = video_feature_dimension[3]
    feat_dim = video_feature_dimension[2]
    text_embed_size = train_dataset.GLOVE_EMBEDDING_SIZE
    answer_vocab_size = None
    
    #############################
    # get word vector dimension
    #############################
    word_matrix = train_dataset.word_matrix
    voc_len = word_matrix.shape[0]
    assert text_embed_size==word_matrix.shape[1]
    
    #############################
    # Parameters
    #############################
    SEQUENCE_LENGTH = args.max_sequence_length
    VOCABULARY_SIZE = train_dataset.n_words
    assert VOCABULARY_SIZE == voc_len
    FEAT_DIM = train_dataset.get_video_feature_dimension()[1:]

    train_iter = train_dataset.batch_iter(args.num_epochs, args.batch_size)


    # Create model directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    if args.task=='Count':
        # add L2 loss
        criterion = nn.MSELoss(size_average=True).cuda()
    elif args.task in ['Action','Trans']:
        from embed_loss import MultipleChoiceLoss
        criterion = MultipleChoiceLoss(num_option=5, margin=1, size_average=True).cuda()
    elif args.task=='FrameQA':
        # add classification loss
        answer_vocab_size = len(train_dataset.ans2idx)
        print('Vocabulary size', answer_vocab_size, VOCABULARY_SIZE)
        criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    
    
    if args.memory_type=='_mrm2s':
        rnn = AttentionTwoStream(args.task, feat_channel, feat_dim, text_embed_size, args.hidden_size,
                             voc_len, args.num_layers, word_matrix, answer_vocab_size = answer_vocab_size, 
                             max_len=args.max_sequence_length)
    else:
        assert 1==2
        
    rnn = rnn.cuda()
    
    #  to directly test, load pre-trained model, replace with your model to test your model
    if args.test==1:
        if args.task == 'Count':
            rnn.load_state_dict(torch.load('./saved_models/Count_concat_fc_mrm2s/rnn-1300-l3.257-a27.942.pkl'))
        elif args.task == 'Action':
            rnn.load_state_dict(torch.load('./saved_models/Action_concat_fc_mrm2s/rnn-0800-l0.137-a84.663.pkl'))
        elif args.task == 'Trans':
            rnn.load_state_dict(torch.load('./saved_models/Trans_concat_fc_mrm2s/rnn-1500-l0.246-a78.068.pkl'))
        elif args.task == 'FrameQA':
            rnn.load_state_dict(torch.load('./saved_models/FrameQA_concat_fc_mrm2s/rnn-4200-l1.233-a69.361.pkl'))
        else:
            assert 1==2, 'Invalid task'
    
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)
    
        
    iter = 0

    if args.task=='Count':

        best_test_loss = 100.0
        best_test_iter = 0.0
        best_val_loss = 100.0
        best_val_iter = 0.0

        # this is a regression problem, predict a value from 1-10
        for batch_chunk in train_iter:

            if args.test==0:
                video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                video_lengths = batch_chunk['video_lengths']
                question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                question_lengths = batch_chunk['question_lengths']
                answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

                data_dict = {}
                data_dict['video_features'] = video_features
                data_dict['video_lengths'] = video_lengths
                data_dict['question_words'] = question_words
                data_dict['question_lengths'] = question_lengths
                data_dict['answers'] = answers

                outputs, targets, predictions = rnn(data_dict, 'Count')
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = rnn.accuracy(predictions, targets.int())
                print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))

            if iter % 100==0:
                rnn.eval()
                with torch.no_grad():

                    if args.test == 0:
                        ##### Validation ######
                        n_iter = len(val_dataset) / args.batch_size
                        losses = AverageMeter()
                        accuracy = AverageMeter()

                        iter_val = 0
                        for batch_chunk in val_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_val % 10 == 0:
                                print('%d/%d' % (iter_val, n_iter))

                            iter_val += 1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                            question_lengths = batch_chunk['question_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

                            # print(question_words)
                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['question_words'] = question_words
                            data_dict['question_lengths'] = question_lengths
                            data_dict['answers'] = answers

                            outputs, targets, predictions = rnn(data_dict, 'Count')
                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets.int())

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))

                        if best_val_loss > losses.avg:
                            best_val_loss = losses.avg
                            best_val_iter = iter

                        print('[Val] iter %d, loss %.3f, acc %.2f, best loss %.3f at iter %d' % (
                            iter, losses.avg, accuracy.avg, best_val_loss, best_val_iter))

                        torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
                                    iter, losses.avg, accuracy.avg)))

                    if 1 == 1:
                        ###### Test ######
                        n_iter = len(test_dataset) / args.batch_size
                        losses = AverageMeter()
                        accuracy = AverageMeter()

                        iter_test = 0
                        for batch_chunk in test_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_test % 10 == 0:
                                print('%d/%d' % (iter_test, n_iter))

                            iter_test+=1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                            question_lengths = batch_chunk['question_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.float32)).cuda()

                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['question_words'] = question_words
                            data_dict['question_lengths'] = question_lengths
                            data_dict['answers'] = answers


                            outputs, targets, predictions = rnn(data_dict, 'Count')
                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets.int())

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))


                        if best_test_loss>losses.avg:
                            best_test_loss = losses.avg
                            best_test_iter = iter

                        print('[Test] iter %d, loss %.3f, acc %.2f, best loss %.3f at iter %d' % (iter,losses.avg,accuracy.avg,best_test_loss,best_test_iter))
                        if args.test==1:
                            exit()

                rnn.train()

            iter += 1
    
    elif args.task in ['Action','Trans']:
        
        best_test_acc = 0.0
        best_test_iter = 0.0
        best_val_acc = 0.0
        best_val_iter = 0.0

        # this is a multiple-choice problem, predict probability of each class
        for batch_chunk in train_iter:

            if args.test == 0:
                video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                video_lengths = batch_chunk['video_lengths']
                candidates = torch.from_numpy(batch_chunk['candidates'].astype(np.int64)).cuda()
                candidate_lengths = batch_chunk['candidate_lengths']
                answers = torch.from_numpy(batch_chunk['answer'].astype(np.int32)).cuda()
                num_mult_choices = batch_chunk['num_mult_choices']


                data_dict = {}
                data_dict['video_features'] = video_features
                data_dict['video_lengths'] = video_lengths
                data_dict['candidates'] = candidates
                data_dict['candidate_lengths'] = candidate_lengths
                data_dict['answers'] = answers
                data_dict['num_mult_choices'] = num_mult_choices


                outputs, targets, predictions = rnn(data_dict, args.task)

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = rnn.accuracy(predictions, targets.long())
                print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))

            if iter % 100==0:
                rnn.eval()
                with torch.no_grad():

                    if args.test == 0:
                        n_iter = len(val_dataset) / args.batch_size
                        losses = AverageMeter()
                        accuracy = AverageMeter()
                        iter_val = 0
                        for batch_chunk in val_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_val % 10 == 0:
                                print('%d/%d' % (iter_val, n_iter))

                            iter_val += 1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            candidates = torch.from_numpy(batch_chunk['candidates'].astype(np.int64)).cuda()
                            candidate_lengths = batch_chunk['candidate_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.int32)).cuda()
                            num_mult_choices = batch_chunk['num_mult_choices']

                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['candidates'] = candidates
                            data_dict['candidate_lengths'] = candidate_lengths
                            data_dict['answers'] = answers
                            data_dict['num_mult_choices'] = num_mult_choices

                            outputs, targets, predictions = rnn(data_dict, args.task)

                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets.long())

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))

                        if best_val_acc < accuracy.avg:
                            best_val_acc = accuracy.avg
                            best_val_iter = iter

                        print('[Val] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (
                                        iter, losses.avg, accuracy.avg, best_val_acc, best_val_iter))
                        torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
                                        iter, losses.avg, accuracy.avg)))

                    if 1==1:
                        n_iter = len(test_dataset) / args.batch_size
                        losses = AverageMeter()
                        accuracy = AverageMeter()
                        iter_test = 0
                        for batch_chunk in test_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_test % 10 == 0:
                                print('%d/%d' % (iter_test, n_iter))

                            iter_test+=1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            candidates = torch.from_numpy(batch_chunk['candidates'].astype(np.int64)).cuda()
                            candidate_lengths = batch_chunk['candidate_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.int32)).cuda()
                            num_mult_choices = batch_chunk['num_mult_choices']
                            #question_word_nums = batch_chunk['question_word_nums']

                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['candidates'] = candidates
                            data_dict['candidate_lengths'] = candidate_lengths
                            data_dict['answers'] = answers
                            data_dict['num_mult_choices'] = num_mult_choices

                            outputs, targets, predictions = rnn(data_dict, args.task)

                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets.long())

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))


                        if best_test_acc < accuracy.avg:
                            best_test_acc = accuracy.avg
                            best_test_iter = iter

                        print('[Test] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (iter,losses.avg,accuracy.avg,best_test_acc,best_test_iter))
                        if args.test == 1:
                            exit()

                rnn.train()

            iter += 1


    elif args.task=='FrameQA':
            
        best_test_acc = 0.0
        best_test_iter = 0.0
        best_val_acc = 0.0
        best_val_iter = 0.0
    
        # this is a multiple-choice problem, predict probability of each class
        for batch_chunk in train_iter:

            if args.test==0:
                video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                video_lengths = batch_chunk['video_lengths']
                question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                question_lengths = batch_chunk['question_lengths']
                answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

                data_dict = {}
                data_dict['video_features'] = video_features
                data_dict['video_lengths'] = video_lengths
                data_dict['question_words'] = question_words
                data_dict['question_lengths'] = question_lengths
                data_dict['answers'] = answers


                outputs, targets, predictions = rnn(data_dict, args.task)

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = rnn.accuracy(predictions, targets)
                print('Train %s iter %d, loss %.3f, acc %.2f' % (args.task,iter,loss.data,acc.item()))


            if iter % 100==0:
                rnn.eval()

                with torch.no_grad():

                    if args.test == 0:
                        losses = AverageMeter()
                        accuracy = AverageMeter()
                        n_iter = len(val_dataset) / args.batch_size

                        iter_val = 0
                        for batch_chunk in val_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_val % 10 == 0:
                                print('%d/%d' % (iter_val, n_iter))

                            iter_val += 1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                            question_lengths = batch_chunk['question_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['question_words'] = question_words
                            data_dict['question_lengths'] = question_lengths
                            data_dict['answers'] = answers

                            outputs, targets, predictions = rnn(data_dict, args.task)

                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets)

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))

                        if best_val_acc < accuracy.avg:
                            best_val_acc = accuracy.avg
                            best_val_iter = iter

                        print('[Val] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (
                                        iter, losses.avg, accuracy.avg, best_val_acc, best_val_iter))
                        torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-l%.3f-a%.3f.pkl' % (
                                        iter, losses.avg, accuracy.avg)))

                    if 1==1:
                        losses = AverageMeter()
                        accuracy = AverageMeter()
                        n_iter = len(test_dataset) / args.batch_size

                        iter_test = 0
                        for batch_chunk in test_dataset.batch_iter(1, args.batch_size, shuffle=False):
                            if iter_test % 10 == 0:
                                print('%d/%d' % (iter_test, n_iter))

                            iter_test+=1

                            video_features = torch.from_numpy(batch_chunk['video_features'].astype(np.float32)).cuda()
                            video_lengths = batch_chunk['video_lengths']
                            question_words = torch.from_numpy(batch_chunk['question_words'].astype(np.int64)).cuda()
                            question_lengths = batch_chunk['question_lengths']
                            answers = torch.from_numpy(batch_chunk['answer'].astype(np.int64)).cuda()

                            data_dict = {}
                            data_dict['video_features'] = video_features
                            data_dict['video_lengths'] = video_lengths
                            data_dict['question_words'] = question_words
                            data_dict['question_lengths'] = question_lengths
                            data_dict['answers'] = answers

                            outputs, targets, predictions = rnn(data_dict, args.task)

                            loss = criterion(outputs, targets)
                            acc = rnn.accuracy(predictions, targets)

                            losses.update(loss.item(), video_features.size(0))
                            accuracy.update(acc.item(), video_features.size(0))
            
            
                        if best_test_acc < accuracy.avg:
                            best_test_acc = accuracy.avg
                            best_test_iter = iter

                        print('[Test] iter %d, loss %.3f, acc %.2f, best acc %.3f at iter %d' % (iter,losses.avg,accuracy.avg,best_test_acc,best_test_iter))
                        if args.test==1:
                            exit()

                rnn.train()

            iter += 1

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int , default=0,
                        help='1-directly test, 0-normal training')
    parser.add_argument('--task', type=str , default='Count',
                         help='[Count, Action, FrameQA, Trans]')
    parser.add_argument('--feat_type', type=str , default='SpTp', 
                         help='[C3D, Resnet, Concat, Tp, Sp, SpTp]')
    parser.add_argument('--save_path', type=str, default='./saved_models/' ,
                        help='path for saving trained models')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2,
                        help='number of layers in lstm')                        
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    

    args = parser.parse_args()
    print(args)
    main(args)