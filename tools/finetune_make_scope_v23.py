"""Optional second, finite round: adapt layer4, select only grouped validation."""
import copy,random
import torch
from tools.make_scope_features_v23 import views,TRAIN_VIEWS,EVAL_VIEWS,key


def refine(encoder,head,rows,cache,val,baseline,measure,weights,initial_score,out):
    train=[r for r in rows if r['split']=='train'];historic=torch.tensor([r['historical'] for r in val['whole'][2]])
    teacher=copy.deepcopy(head).eval()
    for p in encoder.parameters():p.requires_grad=False
    for p in encoder.layer4.parameters():p.requires_grad=True
    optimizer=torch.optim.AdamW([{'params':encoder.layer4.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':.0005}],weight_decay=.02)
    best=initial_score;best_encoder=copy.deepcopy(encoder.state_dict());best_head=copy.deepcopy(head.state_dict());best_val=val
    history=[];best_epoch=0
    # Six epochs maximum; no test data or user photographs enter this procedure.
    for epoch in range(6):
        encoder.eval();head.train();order=list(range(len(train)));random.shuffle(order)
        for first in range(0,len(order),12):
            batch=[train[i] for i in order[first:first+12]];kinds=[random.choice(TRAIN_VIEWS) for _ in batch]
            # Train one random full/masked/mirrored view, validate the actual four-view pipeline.
            inputs=torch.stack([views(r,k)[random.choices(range(4),weights=[.375,.375,.125,.125])[0]] for r,k in zip(batch,kinds)])
            labels=torch.tensor([r['y'] for r in batch]);old=torch.stack([cache['items'][key(r,k)]['feature'] for r,k in zip(batch,kinds)])
            optimizer.zero_grad();feature=torch.nn.functional.normalize(encoder(inputs),dim=1);logits=head(feature)
            loss=torch.nn.functional.cross_entropy(logits,labels,weight=weights,label_smoothing=.04)
            with torch.no_grad():prior=teacher(old).softmax(1)
            loss=loss+.2*torch.nn.functional.kl_div(logits.log_softmax(1),prior,reduction='batchmean')+.1*torch.nn.functional.mse_loss(feature,old)
            loss.backward();optimizer.step()
        encoder.eval();head.eval();current={}
        with torch.inference_mode():
            for kind in EVAL_VIEWS:
                features=[];rs=val[kind][2]
                for first in range(0,len(rs),8):
                    z=encoder(torch.cat([views(r,kind) for r in rs[first:first+8]])).reshape(-1,4,512)
                    features.append(torch.nn.functional.normalize(z[:,:2].mean(1)*.75+z[:,2:].mean(1)*.25,dim=1))
                current[kind]=(torch.cat(features),val[kind][1],rs)
            probs={k:head(v[0]).softmax(1) for k,v in current.items()}
        metrics={k:dict(all=measure(p,current[k][1]),historical=measure(p,current[k][1],historic)) for k,p in probs.items()}
        score=sum(metrics[k]['all']['macro']*dict(whole=.5,phone=.25,upper=.25)[k] for k in metrics)
        guard=all(metrics[k]['historical']['top1']>=baseline[k]['top1']-.02 for k in metrics)
        if guard and score>best:
            best=score;best_epoch=epoch+1;best_encoder=copy.deepcopy(encoder.state_dict());best_head=copy.deepcopy(head.state_dict());best_val=current
        record=dict(epoch=epoch+1,score=score,guard=guard,metrics=metrics);history.append(record)
        import json
        print('FINETUNE',json.dumps(record),flush=True)
        (out/'finetune_progress.json').write_text(json.dumps(dict(best_epoch=best_epoch,history=history)),encoding='utf-8')
    encoder.load_state_dict(best_encoder);head.load_state_dict(best_head);encoder.eval();head.eval()
    return encoder,head,best_val,dict(epochs=6,selected=best_epoch>0,best_epoch=best_epoch,history=history,score=best)
