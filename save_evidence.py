import argparse
import os
import logging
from tqdm import tqdm
import torch
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from save_evidence_data_wiqa import data_wiqa
from save_evidence_data_cosmosqa import data_cosmosqa



def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--task_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--do_lower_case', action='store_true')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_evid_length', type=int, required=True)
    parser.add_argument('--num_paragraph_sents', type=int, required=True)
    parser.add_argument('--num_evidence_sents', type=int, required=True)

    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()

    return args


class EvidenceBert(BertPreTrainedModel):
    def __init__(self, config):
        super(EvidenceBert, self).__init__(config)

        self.roberta = RobertaModel(config)

    def forward(self, features):
        qid = features['qid'][0]

        ante_evid_ids = features['ante_evid_ids'].to(device)
        ante_evid_mask = features['ante_evid_mask'].to(device)
        ante_segment_ids = features['ante_segment_ids'].to(device)

        if 'cons_evid_ids' in features.keys():
            cons_evid_ids = features['cons_evid_ids'].to(device)
            cons_evid_mask = features['cons_evid_mask'].to(device)
            cons_segment_ids = features['cons_segment_ids'].to(device)
        else:
            cons_evid_ids = None
            cons_evid_mask = None
            cons_segment_ids = None

        batch_size, n_evid_sents, seq_length = ante_evid_ids.size()

        if cons_evid_ids is not None:  # wiqa evidence
            evid_ids = torch.stack([ante_evid_ids, cons_evid_ids], dim=0)  # 2 x b x n_evid_sents x max_evid_length
            evid_mask = torch.stack([ante_evid_mask, cons_evid_mask], dim=0)
            segment_ids = torch.stack([ante_segment_ids, cons_segment_ids], dim=0)

            evid_ids = evid_ids.view(-1, seq_length)  # (2 x b x n_evid_sents) x max_evid_length
            evid_mask = evid_mask.view(-1, seq_length)
            segment_ids = segment_ids.view(-1, seq_length)

            output = self.roberta(input_ids=evid_ids, attention_mask=evid_mask)
            output = output[1]  # (2 x b x n_evid_sents) x d_model
            output = output.view(2, batch_size, n_evid_sents, -1)  # 2 x b x n_evid_sents x d_model
            return qid, output[0].squeeze(0), output[1].squeeze(0)

        else:  # cosmosqa evidence
            evid_ids = ante_evid_ids.view(-1, seq_length)  # (b x n_evid_sents) x max_evid_length
            evid_mask = ante_evid_mask.view(-1, seq_length)
            segment_ids = ante_segment_ids.view(-1, seq_length)

            output = self.roberta(input_ids=evid_ids, attention_mask=evid_mask)
            output = output[1]  # (2 x b x n_evid_sents) x d_model

            output = output.view(batch_size, n_evid_sents, -1)  # b x n_evid_sents x d_model

            return qid, output.squeeze(0), None

if __name__ == '__main__':
    opt = parse_opt()

    assert opt.batch_size == 1, 'batch_size should be 1.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        raise Exception('n_gpu > 1 is not allowed.')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = RobertaConfig.from_pretrained(opt.model_name_or_path)
    evidbert = EvidenceBert.from_pretrained(opt.model_name_or_path,
                                            config=config).to(device)
    evidbert = evidbert.to(device)

    output = {}
    with torch.no_grad():
        if opt.do_eval:
            if opt.task_name == "wiqa":
                loader = data_wiqa(opt, "eval")
            elif opt.task_name == "cosmosqa":
                loader = data_cosmosqa(opt, "eval")

            i = 0
            all = len(loader)
            for data in tqdm(loader, desc="saving eval evids"):
                i += 1
                # print('{}/{}'.format(i, all))
                evidbert.eval()
                qid, ante_feats, cons_feats = evidbert(data)
                ante_feats = ante_feats.cpu().data.numpy()  # n_evid_sents x d_model
                # print('ante_feats.shape', ante_feats.shape, type(ante_feats), ante_feats.dtype)
                if cons_feats is not None:
                    cons_feats = cons_feats.cpu().data.numpy()  # n_evid_sents x d_model
                    # print("cons_feats.shape", cons_feats.shape, type(cons_feats), cons_feats.dtype)
                    output[qid] = [ante_feats, cons_feats]  # list of np arrays
                else:
                    output[qid] = [ante_feats]

            if not os.path.isdir(os.path.join(opt.data_dir, 'try_feats_roberta_large')):
                os.mkdir(os.path.join(opt.data_dir, 'try_feats_roberta_large'))
            save_name = os.path.join(opt.data_dir, 'try_feats_roberta_large', 'dev_ante_cons_feats.pt')
            torch.save(output, save_name)

        if opt.do_predict:
            if opt.task_name == "wiqa":
                loader = data_wiqa(opt, "test")
            elif opt.task_name == "cosmosqa":
                loader = data_cosmosqa(opt, "test")

            i = 0
            all = len(loader)
            for data in tqdm(loader, desc="saving test evids"):
                i += 1
                # print('{}/{}'.format(i, all))
                evidbert.eval()
                qid, ante_feats, cons_feats = evidbert(data)
                ante_feats = ante_feats.cpu().data.numpy()  # n_evid_sents x d_model
                # print('ante_feats.shape', ante_feats.shape, type(ante_feats), ante_feats.dtype)
                if cons_feats is not None:
                    cons_feats = cons_feats.cpu().data.numpy()  # n_evid_sents x d_model
                    # print("cons_feats.shape", cons_feats.shape, type(cons_feats), cons_feats.dtype)
                    output[qid] = [ante_feats, cons_feats]  # list of np arrays
                else:
                    output[qid] = [ante_feats]

            if not os.path.isdir(os.path.join(opt.data_dir, 'try_feats_roberta_large')):
                os.mkdir(os.path.join(opt.data_dir, 'try_feats_roberta_large'))
            save_name = os.path.join(opt.data_dir, 'try_feats_roberta_large', 'test_ante_cons_feats.pt')
            torch.save(output, save_name)

        if opt.do_train:
            if opt.task_name == "wiqa":
                loader = data_wiqa(opt, "train")
            elif opt.task_name == "cosmosqa":
                loader = data_cosmosqa(opt, "train")

            if not os.path.isdir(os.path.join(opt.data_dir, 'try_feats_roberta_large', 'train_ante_cons_feats')):
                os.mkdir(os.path.join(opt.data_dir, 'try_feats_roberta_large', 'train_ante_cons_feats'))

            i = 0
            for data in tqdm(loader, desc="saving train evids"):
                i += 1
                # print('i={}'.format(i))
                evidbert.eval()
                qid, ante_feats, cons_feats = evidbert(data)
                save_name = os.path.join(opt.data_dir, 'try_feats_roberta_large', 'train_ante_cons_feats',
                                         '{}_ante_cons_feats.pt'.format(qid))
                ante_feats = ante_feats.cpu().data.numpy()  # n_evid_sents x d_model
                if cons_feats is not None:
                    cons_feats = cons_feats.cpu().data.numpy()  # n_evid_sents x d_model
                    torch.save([ante_feats, cons_feats], save_name)
                else:
                    torch.save([ante_feats], save_name)
                print('Saved {}.'.format(save_name))

