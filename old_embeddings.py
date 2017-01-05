import codecs

########## Emebeddings
def Compare_ANEW(file_opinion):
    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            dim_w = h_word2dim[w]
            if dim_w >= N:
                continue
            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    topn, p = 1500, 1
    t_r, t_rho, t_tau = [], [], []
    for w in h_word2valence:
        valence = h_word2valence[w]
        dominance = h_word2dominance[w]
        arousal = h_word2arousal[w]

        # closest words
        #t_closest_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_closest_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_closest_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]

        h_threeD = {}
        for v in list(h_word2arousal):
            h_arousal = abs(arousal-h_word2arousal[v])**p
            h_valence = abs(valence-h_word2valence[v])**p
            h_dominance = abs(dominance-h_word2dominance[v])**p
            h_threeD[v] = (h_arousal + h_valence + h_dominance)**(1/p)

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -m_embed[h_word2dim[w],
                                                           h_word2dim[w2]]))
        # t_closest_threeD = t_closest_threeD[::60]

        # random
        #t_random_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]
        #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda v:sqrt(abs(arousal-h_word2arousal[v])**2 + abs(valence-h_word2valence[v])**2 + abs(dominance-h_word2dominance[v])**2 ) )[0:topn]

        #t_valence_ordered = [h_word2valence[w2] for w2 in t_closest_valence]
        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-m_embed[h_word2dim[w], h_word2dim[w2]]
                        for w2 in t_closest_threeD]

        t_r.append(scipy.stats.stats.pearsonr(t_threeD_ordered,
                                              t_NN_ordered)[0])
        t_rho.append(scipy.stats.stats.spearmanr(t_threeD_ordered,
                                                 t_NN_ordered)[0])
        t_tau.append(scipy.stats.stats.kendalltau(t_threeD_ordered,
                                                  t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean pearson r =', sum(t_r)/len(t_r))
    print('Mean spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean kendall tau =', sum(t_tau)/len(t_tau))


def Compare_SimLexANEW(file_simlex, file_opinion):

    h_convert_POS = {'A': 'ADJ', 'N': 'N', 'V': 'V'}
    h_SimLex, h_voc = {}, {}
    for line_no, line in enumerate(codecs.open(file_simlex, 'r', 'utf-8')):
        if line_no == 0:
            continue
        t_line = line.split('\t')
        w1 = t_line[0]
        w2 = t_line[1]
        POS = t_line[2]
        simlex999 = t_line[3]
        bool_sim333 = t_line[8]

        POS = h_convert_POS[POS]
        w1, w2 = w1+'___'+POS, w2+'___'+POS
        simlex999 = float(simlex999)
        h_SimLex[w1+'###'+w2] = simlex999
        h_voc[w1] = 1
        h_voc[w2] = 1

    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            dim_w = h_word2dim[w]
            if dim_w >= N:
                continue
            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    t_opinion_ordered, t_simlex_ordered = [], []
    for tw in h_SimLex:
        w1, w2 = tw.split('###')
        if w1 in h_word2valence and w2 in h_word2valence:
            t_opinion_ordered.append(abs(h_word2valence[w1]
                                         - h_word2valence[w2]))
            t_simlex_ordered.append(h_SimLex[tw])

        #valence, dominance, arousal = h_word2valence[w],h_word2dominance[w],h_word2arousal[w]

    print('Pearson r SimLex/ANEW = ', pearsonr(t_opinion_ordered,
                                               t_simlex_ordered)[0])
    print('Spearman rho SimLex/ANEW = ', spearmanr(t_opinion_ordered,
                                                   t_simlex_ordered)[0])
    print('Kendall tau SimLex/ANEW = ', kendalltau(t_opinion_ordered,
                                                   t_simlex_ordered)[0])


def Compare_SentiWN(file_opinion):

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    print('nb mots de sentiWN ds embed ', len(h_word2objective))

    topn, p = 180, 1
    t_r, t_rho, t_tau = [], [], []
    for w in random.sample(list(h_word2positive), 2000):
        objective = h_word2objective[w]
        positive = h_word2positive[w]
        negative = h_word2negative[w]

        # closest words

        #h_threeD = { v:(abs(positive-h_word2positive[v])**p + abs(negative-h_word2negative[v])**p )**(1/p) for v in list(h_word2positive) }
        h_threeD = {v: abs(negative-h_word2negative[v])
                    for v in h_word2positive}

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -m_embed[h_word2dim[w],
                                                           h_word2dim[w2]]))
        t_closest_threeD = t_closest_threeD[::topn]

        # random
        #t_random_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]
        #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda v:sqrt(abs(arousal-h_word2arousal[v])**2 + abs(valence-h_word2valence[v])**2 + abs(dominance-h_word2dominance[v])**2 ) )[0:topn]

        #t_valence_ordered = [h_word2valence[w2] for w2 in t_closest_valence]
        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-m_embed[h_word2dim[w], h_word2dim[w2]]
                        for w2 in t_closest_threeD]

        t_r.append(pearsonr(t_threeD_ordered, t_NN_ordered)[0])
        t_rho.append(spearmanr(t_threeD_ordered, t_NN_ordered)[0])
        t_tau.append(kendalltau(t_threeD_ordered, t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean Pearson r =', sum(t_r)/len(t_r))
    print('Mean Spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean Kendall tau =', sum(t_tau)/len(t_tau))


def Compare_SimLexSentiWN(file_simlex, file_opinion):

    h_convert_POS = {'A': 'ADJ', 'N': 'N', 'V': 'V'}
    h_SimLex, h_voc = {}, {}
    for line_no, line in enumerate(codecs.open(file_simlex, 'r', 'utf-8')):
        if line_no == 0:
            continue
        t_line = line.split('\t')
        w1 = t_line[0]
        w2 = t_line[1]
        POS = t_line[2]
        simlex999 = t_line[3]
        bool_sim333 = t_line[8]

        POS = h_convert_POS[POS]
        w1, w2 = w1+'___'+POS, w2+'___'+POS
        simlex999 = float(simlex999)
        h_SimLex[w1+'###'+w2] = simlex999
        h_voc[w1] = 1
        h_voc[w2] = 1

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    print('nb mots de sentiWN ds SimLex ', len(h_word2objective))

    t_opinion_ordered, t_simlex_ordered = [], []
    for tw in h_SimLex:  # sorted(list(h_SimLex),key=h_SimLex.get)[::10]:
        w1, w2 = tw.split('###')
        if w1 in h_word2objective and w2 in h_word2objective:
            t_opinion_ordered.append(abs(h_word2negative[w1]
                                         - h_word2negative[w2]))
            t_simlex_ordered.append(h_SimLex[tw])

    print('size of intersection: ', len(t_opinion_ordered))
    print('Pearson r SimLex/SentiWN = ', pearsonr(t_opinion_ordered,
                                                  t_simlex_ordered)[0])
    print('Spearman rho SimLex/SentiWN = ', spearmanr(t_opinion_ordered,
                                                      t_simlex_ordered)[0])
    print('Kendall tau SimLex/SentiWN = ', kendalltau(t_opinion_ordered,
                                                      t_simlex_ordered)[0])


def Compare_ANEWSentiWN(file_anew, file_sentiwn):

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                dim_w = h_word2dim[w]
                if dim_w >= N:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_anew) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    print('nb mots de sentiWN ds ANEW ', len(h_word2valence))

    t_word = sorted(list(h_word2valence), key=lambda w: h_word2valence[w])
    t_opinion1_ordered = [h_word2valence[w] for w in t_word]
    t_opinion2_ordered = [h_word2positive[w]-h_word2negative[w]
                          for w in t_word]

    print('size of intersection: ', len(t_opinion1_ordered))
    print('Pearson r SimLex/SentiWN = ', pearsonr(t_opinion1_ordered,
                                                  t_opinion2_ordered)[0])
    print('Spearman rho SimLex/SentiWN = ', spearmanr(t_opinion1_ordered,
                                                      t_opinion2_ordered)[0])
    print('Kendall tau SimLex/SentiWN = ', kendalltau(t_opinion1_ordered,
                                                      t_opinion2_ordered)[0])


def Compare_W2V_ANEW(file_anew):
    model_w2v = models.Word2Vec.load_word2vec_format('/home/vincent/Data/Data/Google_Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

    h_word2valence, h_word2arousal, h_word2dominance = {}, {}, {}
    with open(file_anew) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            l = l.rstrip('\r\n')
            w, nb, valence, _, arousal, _, dominance, _ = l.split('\t')
            if w+'___N' in h_word2dim:
                w += '___N'
            elif w+'___V' in h_word2dim:
                w += '___V'
            elif w+'___A' in h_word2dim:
                w += '___A'
            else:
                continue

            h_word2valence[w] = float(valence)
            h_word2dominance[w] = float(dominance)
            h_word2arousal[w] = float(arousal)

    topn, p = 1500, 1
    t_r, t_rho, t_tau = [], [], []
    for w in h_word2valence:
        valence = h_word2valence[w]
        dominance = h_word2dominance[w]
        arousal = h_word2arousal[w]

        h_threeD = {}
        for v in list(h_word2arousal):
            h_arousal = abs(arousal-h_word2arousal[v])**p
            h_valence = abs(valence-h_word2valence[v])**p
            h_dominance = abs(dominance-h_word2dominance[v])**p
            h_threeD[v] = (h_arousal + h_valence + h_dominance)**(1/p)

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -m_embed[h_word2dim[w],
                                                           h_word2dim[w2]]))
        # t_closest_threeD = t_closest_threeD[::60]

        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-m_embed[h_word2dim[w], h_word2dim[w2]]
                        for w2 in t_closest_threeD]

        t_r.append(scipy.stats.stats.pearsonr(t_threeD_ordered,
                                              t_NN_ordered)[0])
        t_rho.append(scipy.stats.stats.spearmanr(t_threeD_ordered,
                                                 t_NN_ordered)[0])
        t_tau.append(scipy.stats.stats.kendalltau(t_threeD_ordered,
                                                  t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean pearson r =', sum(t_r)/len(t_r))
    print('Mean spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean kendall tau =', sum(t_tau)/len(t_tau))


def Compare_W2V_SentiWN(file_opinion):
    from gensim import models
    model_w2v = models.Word2Vec.load_word2vec_format('/home/vincent/Data/Data/Google_Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

    h_word2positive, h_word2negative, h_word2objective = {}, {}, {}
    with open(file_opinion) as f:
        for l in f:
            if l.startswith('Word') or l.startswith('#'):
                continue
            t, _, pos, neg, t_w, _ = l.rstrip('\r\n').split('\t')

            for w in t_w.split(' '):
                w = re.sub('#.+', '', w)

                if t == 'n' and w+'___N' in h_word2dim:
                    w += '___N'
                elif t == 'v' and w+'___V' in h_word2dim:
                    w += '___V'
                elif t == 'r' and w+'___ADV' in h_word2dim:
                    w += '___ADV'
                elif t == 'a' and w+'___A' in h_word2dim:
                    w += '___A'
                else:
                    continue

                if w not in model_w2v:
                    continue

                h_word2positive[w] = float(pos)
                h_word2negative[w] = float(neg)
                h_word2objective[w] = 1 - (float(pos)+float(neg))

    print('nb mots de sentiWN ds W2V ', len(h_word2objective))

    topn, p = 180, 1
    t_r, t_rho, t_tau = [], [], []
    for w in random.sample(list(h_word2positive), 2000):
        objective = h_word2objective[w]
        positive = h_word2positive[w]
        negative = h_word2negative[w]

        # closest words

        #h_threeD = { v:(abs(positive-h_word2positive[v])**p + abs(negative-h_word2negative[v])**p )**(1/p) for v in list(h_word2positive) }
        h_threeD = {v: abs(objective-h_word2objective[v])
                    for v in h_word2positive}

        t_closest_threeD = sorted(list(h_threeD),
                                  key=lambda w2: (h_threeD[w2],
                                                  -model_w2v.similarity(w, w2)))
        t_closest_threeD = t_closest_threeD[::topn]

        # random
        #t_random_valence = sorted(h_word2valence.keys(),key=lambda v:abs(valence-h_word2valence[v]))[0:topn]
        #t_random_dominance = sorted(h_word2dominance.keys(),key=lambda v:abs(dominance-h_word2dominance[v]))[0:topn]
        #t_random_arousal = sorted(h_word2arousal.keys(),key=lambda v:abs(arousal-h_word2arousal[v]))[0:topn]
        #t_random_threeD = sorted(h_word2arousal.keys(),key=lambda v:sqrt(abs(arousal-h_word2arousal[v])**2 + abs(valence-h_word2valence[v])**2 + abs(dominance-h_word2dominance[v])**2 ) )[0:topn]

        #t_valence_ordered = [h_word2valence[w2] for w2 in t_closest_valence]
        t_threeD_ordered = [h_threeD[w2] for w2 in t_closest_threeD]

        t_NN_ordered = [-model_w2v.similarity(w,w2) for w2 in t_closest_threeD]

        t_r.append(pearsonr(t_threeD_ordered, t_NN_ordered)[0])
        t_rho.append(spearmanr(t_threeD_ordered, t_NN_ordered)[0])
        t_tau.append(kendalltau(t_threeD_ordered, t_NN_ordered)[0])

    print('number of words processed: ', len(t_r))
    print('Mean Pearson r =', sum(t_r)/len(t_r))
    print('Mean Spearman rho =', sum(t_rho)/len(t_rho))
    print('Mean Kendall tau =', sum(t_tau)/len(t_tau))



def BuildQREL_from_NRC_W2V(file_opinion, model_w2v):
     Info('NRC with W2V')

     h_word2class, h_class2word = defaultdict(lambda:[]), defaultdict(lambda:[])
     # ordre des classes: anger,anticipation, disgust, fear,joy, negative, positive, sadness, surprise, trust
     with open(file_opinion) as f:
         for l in f:
             t_l = l.rstrip('\r\n').split('\t')
             if len(t_l)==3:
                 w,v = t_l[0],int(t_l[2])

                 if w+'_N' in model_w2v: w+='_N'
                 elif w+'_V' in model_w2v: w+='_V'
                 elif w+'_A' in model_w2v: w+='_A'
                 else:
                     # si on utilise les vecteur de Google News
                     if w not in model_w2v: continue


                 #if t_l[1] not in ['positive','negative']: continue
                 if t_l[1] not in ['sadness']: continue
                 h_word2class[w].append(str(v))
                 h_class2word[t_l[1]].append(w)

     f_qrel_anger = codecs.open(args.prefix+'.NRC_all_dim.qrel','w')
     qid, h_qid2w = 1, {}
     h_seen = {}
     for w in list(h_word2class)[0:5000]: # 1000 requetes suffisent
         vector = ''.join(h_word2class[w])
         if w in h_seen or vector == '0': continue
         #if w in h_seen or vector == '0000000000': continue
         #if w in h_seen or vector == '00': continue
         t_pos_w = [w]
         t_identical = [ w2 for w2 in h_word2class if ''.join(h_word2class[w2])==vector and not w2==w and not w2 in h_seen ]
         #t_pos_w += random.sample(t_identical,min(len(t_identical),1))
         for w2 in t_pos_w: h_seen[w2]=1
         t_neg_w = []
         #s_anti_w = set([ w2 for w2 in h_word2class if not ''.join(h_word2class[w2]) == vector and not ''.join(h_word2class[w2])=='00'])
         #for t_elem in MostSimilar([w],m_embed,h_word2dim, topn=200):#, possible_tag='same_as_query'):
         #    nn,score = t_elem
         #    if nn in s_anti_w:
         #        t_neg_w.append(nn)
         #        if len(t_neg_w)>0: break
         #print(w, '  ;  neg : '+' '.join(t_neg_w))


         for nn in ( w2 for w2 in h_word2class if vector == (''.join(h_word2class[w2])) ):
             if nn in t_pos_w: continue
             f_qrel_anger.write(str(qid)+' 0 '+nn+' 1\n')

         h_qid2w[qid]=(t_pos_w,t_neg_w)
         qid+=1
     f_qrel_anger.close()
     return h_qid2w, h_word2class

