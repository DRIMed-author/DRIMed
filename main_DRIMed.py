import os
import time
import dill
import torch
import argparse
import importlib
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict

from src.utils.metrics import set_seed, llprint, multi_label_metric, ddi_rate_score
from src.utils.data_loader import load_dataset


# All model classes expose the same interface:
#   Model(vocab_size, ehr_adj, ddi_adj, emb_dim, device) with forward(input) -> (result, ddi_loss)
MODEL_REGISTRY = {
    # main model
    'DRIMed': 'src.models.DRIMed',
    # ablation variants
    'DRIMed_wo_DARC': 'src.models.DRIMed_wo_DARC',
    'DRIMed_wo_HMR': 'src.models.DRIMed_wo_HMR',
    'DRIMed_wo_IIM': 'src.models.DRIMed_wo_IIM',
    'DRIMed_wo_MKG': 'src.models.DRIMed_wo_MKG',
    'DRIMed_wo_RCE': 'src.models.DRIMed_wo_RCE',
    # baselines
    'CausalMed': 'src.Baselines.CausalMed',
    'COGNet': 'src.Baselines.COGNet',
    'GAMENet': 'src.Baselines.GAMENet',
    'HyperMed': 'src.Baselines.HyperMed',
    'LEAP': 'src.Baselines.LEAP',
    'MoleRec': 'src.Baselines.MoleRec',
    'MrDTR': 'src.Baselines.MrDTR',
    'RAVSNet': 'src.Baselines.RAVSNet',
    'RETAIN': 'src.Baselines.RETAIN',
    'SafeDrug': 'src.Baselines.SafeDrug',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DRIMed', choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--target_ddi', type=float, default=0.06)
    parser.add_argument('--kp', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--saved_dir', type=str, default='saved')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default='', help='checkpoint filename under saved_dir')
    return parser.parse_args()


def eval_model(model, data_eval, voc_size, device, args):
    model.eval()
    smm_record = []
    ja, prauc, avg_f1 = [], [], []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        if len(input) < 2:
            continue
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[:adm_idx + 1])
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= args.threshold] = 1
            y_pred_tmp[y_pred_tmp < args.threshold] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label.append(sorted(np.where(y_pred_tmp == 1)[0]))
            visit_cnt += 1
            med_cnt += len(y_pred_label[-1])

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, _, _, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, path=args.ddi_path)
    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt


def main():
    args = parse_args()
    set_seed(3407)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    data = load_dataset(args.dataset, args.data_dir)
    data_train, data_eval, data_test = data['train'], data['eval'], data['test']
    voc_size = data['voc_size']
    ehr_adj = data['ehr_adj']
    ddi_adj = data['ddi_adj']
    args.ddi_path = data['ddi_path']

    os.makedirs(args.saved_dir, exist_ok=True)

    model_cls = importlib.import_module(MODEL_REGISTRY[args.model]).Model
    model = model_cls(voc_size, ehr_adj, ddi_adj, emb_dim=args.emb_dim, device=device).to(device)

    if args.test:
        assert args.resume, '--test requires --resume <checkpoint>'
        model.load_state_dict(torch.load(os.path.join(args.saved_dir, args.resume), map_location=device), strict=False)
        eval_model(model, data_test, voc_size, device, args)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def criterion(preds):
        result, loss_ddi = preds
        loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
        loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

        prob = F.sigmoid(result).detach().cpu().numpy()[0]
        prob[prob >= args.threshold] = 1
        prob[prob < args.threshold] = 0
        y_label = np.where(prob == 1)[0]
        current_ddi_rate = ddi_rate_score([[y_label]], path=args.ddi_path)

        if current_ddi_rate <= args.target_ddi:
            loss = args.alpha * loss_bce + (1 - args.alpha) * loss_multi
        else:
            beta = 1 - min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
            loss = (args.alpha * loss_bce + (1 - args.alpha) * loss_multi) + beta * loss_ddi
        return loss

    best_ja = 0.0
    best_name = ''
    for epoch in range(args.epochs):
        tic = time.time()
        model.train()
        mean_loss = 0.0
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue
            loss = 0
            for idx, adm in enumerate(input):
                seq_input = input[:idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for i, item in enumerate(adm[2]):
                    loss_multi_target[0][i] = item

                loss = criterion(model(seq_input))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.item()
            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        ddi_rate, ja, prauc, avg_f1, avg_med = eval_model(model, data_eval, voc_size, device, args)
        print('\nepoch {}: loss {:.4f}, time {:.1f}s'.format(epoch, mean_loss / max(len(data_train), 1), time.time() - tic))

        name = 'Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, args.target_ddi, ja, ddi_rate)
        if ja > best_ja:
            best_ja = ja
            best_name = name
            torch.save(model.state_dict(), os.path.join(args.saved_dir, name))

    print('best:', best_name, 'best_ja:', best_ja)


if __name__ == '__main__':
    main()