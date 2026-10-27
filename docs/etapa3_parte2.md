# Etapa 3, parte 2: Geometria epipolar

## 1. Objetivo

A primeira parte da Etapa 3 produziu, para cada detector e cada par de quadros consecutivos (`matching.strategy: sequential`), as correspondências filtradas pelo teste da razão de Lowe (`matches_lowe.npy`). Esta segunda parte usa essas correspondências para recuperar a geometria entre as duas vistas: (i) estimar as matrizes fundamental $F$ e essencial $E$ com um estimador robusto; (ii) recuperar o movimento relativo $(R, \mathbf{t})$ da câmera 2 em relação à câmera 1; (iii) visualizar as linhas epipolares como verificação qualitativa; e (iv) triangular uma nuvem de pontos 3D inicial por par, que serve de ponto de partida para a reconstrução das etapas seguintes. O código está em `src/epipolar/` (`estimate.py`, `pose.py`, `visualize.py`, `triangulate.py`, orquestrados por `__init__.py` e `two_view.py`).

## 2. Matrizes fundamental e essencial

Um ponto $\mathbf{x}_1$ (em coordenadas homogêneas de pixel) no quadro 1 define um raio no espaço; visto pela câmera 2, esse raio projeta-se numa reta, a linha epipolar, sobre a qual deve estar o correspondente $\mathbf{x}_2$. A restrição epipolar é

$$\mathbf{x}_2^\top F\,\mathbf{x}_1 = 0,$$

com $F$ de posto 2 e 7 graus de liberdade. Removendo a influência da lente pelas intrínsecas $K$, obtém-se a matriz essencial $E = K^\top F K = [\mathbf{t}]_\times R$, que depende apenas do movimento entre as câmeras (5 graus de liberdade).

**Intrínsecas.** Como a câmera do celular não foi calibrada, `geometry.intrinsics` fica em `null` e $K$ é aproximada a partir da resolução $w \times h$ do primeiro quadro de cada par (`_intrinsics_matrix` em `__init__.py`): $f_x = f_y = \max(w, h)$, $c_x = w/2$, $c_y = h/2$, sem distorção. Valores explícitos no YAML substituem esses padrões. A matriz usada é salva por par (`K.npy`) e também em `results/<id>/K.npy`.

**Estimação robusta.** Para cada par com pelo menos 8 correspondências, `estimate.py` chama `cv2.findFundamentalMat` (pontos em pixels) e `cv2.findEssentialMat` (pontos em pixels e $K$), ambos com o método indicado em `geometry.ransac.method`, resolvido com `getattr(cv2, ...)`. O padrão é `USAC_MAGSAC`, variante do RANSAC que pondera os resíduos por um modelo de ruído marginalizado e é menos sensível ao limiar fixo. Os parâmetros vêm do YAML: `threshold_px = 1.0` (distância máxima, em pixels, de um inlier à sua linha epipolar), `confidence = 0.999` e `max_iters = 5000`. Pares com menos de 8 correspondências, ou para os quais o estimador falha, recebem matrizes preenchidas com NaN e máscara vazia.

**Máscara de inliers.** Uma correspondência só é considerada inlier se for aceita pelos dois modelos: `inlier_mask = mask_F AND mask_E`. Isso descarta pontos coerentes com $F$ mas incompatíveis com as intrínsecas assumidas (e vice-versa).

**Erro de Sampson.** Para os inliers calcula-se o erro de Sampson, aproximação de primeira ordem da distância geométrica à variedade epipolar, em pixels:

$$d_S = \frac{|\mathbf{x}_2^\top F \mathbf{x}_1|}{\sqrt{(F\mathbf{x}_1)_1^2 + (F\mathbf{x}_1)_2^2 + (F^\top\mathbf{x}_2)_1^2 + (F^\top\mathbf{x}_2)_2^2}}.$$

Correspondências com denominador numericamente nulo recebem NaN.

## 3. Recuperação de $R$ e $\mathbf{t}$

A decomposição de $E$ por SVD fornece quatro soluções candidatas: duas rotações ($R_a$, $R_b$) combinadas com $\pm\mathbf{t}$. Apenas uma delas coloca os pontos triangulados à frente das duas câmeras. `pose.py` usa `cv2.recoverPose(E, pts1, pts2, K, mask)`, que triangula os inliers para cada candidata e escolhe a que passa no **teste de quiralidade** (profundidade positiva nas duas câmeras) para o maior número de pontos.

- A entrada é a `inlier_mask` da seção 2; a saída `pose_mask` é a interseção dessa máscara com os pontos aprovados pelo teste de quiralidade, e `n_pose_inliers` é sua contagem.
- O par é ignorado (métricas NaN, nenhum arquivo gravado) se $E$ não for finita, se houver menos de 5 inliers (amostra mínima da matriz essencial) ou se nenhum ponto passar no teste.
- $\mathbf{t}$ só é determinável **a menos de escala**: $E$ é homogênea, logo qualquer múltiplo de $\mathbf{t}$ gera a mesma restrição. O vetor é normalizado para $\|\mathbf{t}\| = 1$ e representa apenas a direção da translação.
- A magnitude da rotação é reportada como o ângulo do vetor de Rodrigues: $\theta = \|\mathrm{rodrigues}(R)\|$, em graus (`rot_deg`).

## 4. Linhas epipolares

Dado $F$, a linha epipolar de $\mathbf{x}_1$ na imagem 2 é $\mathbf{l}' = F\mathbf{x}_1$, e a de $\mathbf{x}_2$ na imagem 1 é $\mathbf{l} = F^\top\mathbf{x}_2$, cada uma na forma $(a, b, c)$ com $ax + by + c = 0$. `visualize.py` sorteia, sem reposição, `visualize.epilines.n_lines = 20` inliers usando um gerador inicializado com `run.seed` (reprodutível) e calcula ambas as linhas com `cv2.computeCorrespondEpilines`. A figura coloca as duas imagens lado a lado, na mesma altura, com largura total de 12 polegadas a `dpi = 150`; cada correspondência tem uma cor própria, e na imagem 1 desenham-se as linhas induzidas pelos pontos da imagem 2 (e vice-versa), com o ponto correspondente marcado. Se a estimação estiver correta, cada ponto cai sobre a linha da sua cor, e as linhas convergem para o epipolo. Pares com menos de 8 inliers ou $F$ não finita não geram figura. Caminho: `results/<id>/figures/epilines/<detector>/<par>.png`.

## 5. Triangulação da nuvem inicial

Fixando a câmera 1 como referencial do mundo, as matrizes de projeção são

$$P_1 = K\,[\,I \mid \mathbf{0}\,], \qquad P_2 = K\,[\,R \mid \mathbf{t}\,].$$

`triangulate.py` aplica `cv2.triangulatePoints(P1, P2, x1, x2)` às correspondências de `pose_mask` (triangulação linear por DLT), obtendo pontos homogêneos $\tilde{X} = (X, Y, Z, W)$, desomogeneizados por $\mathbf{X} = (X/W,\, Y/W,\, Z/W)$. Mantêm-se apenas os pontos finitos com profundidade positiva nas duas câmeras: $Z_1 = X_z > 0$ e $Z_2 = (R\mathbf{X} + \mathbf{t})_z > 0$. A fração mantida é `cheirality_frac`.

Para cada ponto calcula-se o **ângulo de paralaxe** entre os raios $\mathbf{X} - C_1$ e $\mathbf{X} - C_2$, com $C_1 = \mathbf{0}$ e $C_2 = -R^\top\mathbf{t}$; a mediana é reportada em `parallax_med_deg`. Paralaxe pequena indica pontos mal condicionados em profundidade. A cor de cada ponto é a do pixel mais próximo de $\mathbf{x}_1$ no quadro 1 (convertida de BGR para RGB).

Saídas por par: `points3d.npy` ($N \times 3$), `colors.npy` ($N \times 3$, `uint8`) e `cloud.ply`. Como $\|\mathbf{t}\| = 1$, a nuvem está numa escala arbitrária, e as nuvens de pares diferentes não estão no mesmo referencial nem na mesma escala; a escala métrica será fixada em etapa posterior a partir de um objeto de dimensão conhecida.

## 6. Saídas e métricas

Para cada par, em `results/<id>/epipolar/<detector>/<quadro1>__<quadro2>/`:

| Arquivo | Conteúdo |
|---|---|
| `F.npy`, `E.npy`, `K.npy` | matrizes $3\times3$ (NaN em caso de falha) |
| `inlier_mask.npy` | máscara booleana $F \wedge E$ sobre `matches_lowe` |
| `sampson_errors.npy` | erro de Sampson dos inliers (px) |
| `R.npy`, `t.npy`, `pose_mask.npy` | pose relativa ($\|\mathbf{t}\|=1$) e inliers que passam na quiralidade |
| `points3d.npy`, `colors.npy`, `cloud.ply` | nuvem triangulada e suas cores |

As métricas de todos os pares são agregadas em `artifacts/<id>/epipolar_metrics.csv`, uma linha por par (chave `id = <detector>/<par>`), com as colunas: `id`, `detector`, `pair`, `n_matches` (correspondências de Lowe), `n_inliers` (inliers $F \wedge E$), `rot_deg`, `t_x`, `t_y`, `t_z`, `n_pose_inliers`, `n_points3d`, `parallax_med_deg` e `cheirality_frac`. Ao final de cada detector, o log registra o número de pares, o número de pares com pose recuperada e a mediana de cada métrica. As figuras ficam em `results/<id>/figures/epilines/<detector>/<par>.png`.

## 7. Resultados

Execução de `python main.py` com o `config.yaml` deste repositório sobre `3d_image_video.MOV` (retrato, quadros salvos em 1920×3413): a extração selecionou 101 quadros pela meta de sobreposição de 0,70, a etapa de qualidade manteve 61 e cada detector gerou 60 pares consecutivos. Valores de `artifacts/config/epipolar_metrics.csv` e `artifacts/config/matching_metrics.csv`; as medianas das colunas de pose e triangulação consideram apenas os pares com pose recuperada.

| Detector | Pares | Com pose | Correspondências (Lowe) | Inliers $F \wedge E$ | Inliers de pose | Rotação (°) | Pontos 3D por par | Pontos 3D (total) | Paralaxe (°) | Fração quiral | Sampson (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AKAZE | 60 | 55 | 508,0 | 161,5 | 177,0 | 7,66 | 151,0 | 13 180 | 2,03 | 1,00 | 0,41 |
| ORB | 60 | 56 | 241,5 | 73,5 | 55,0 | 7,77 | 52,0 | 6 535 | 1,96 | 1,00 | 0,40 |
| SIFT | 60 | 56 | 173,5 | 46,0 | 44,0 | 7,96 | 40,0 | 3 773 | 2,16 | 1,00 | 0,38 |

Medianas por par, exceto as colunas "Pares", "Com pose" e "Pontos 3D (total)". A mediana dos inliers de pose do AKAZE fica acima da de inliers $F \wedge E$ apenas porque as duas medianas são calculadas sobre conjuntos de pares diferentes; restrita aos pares com pose, a mediana de inliers $F \wedge E$ é 195 (AKAZE), 79 (ORB) e 53,5 (SIFT), e em nenhum par há mais inliers de pose que inliers $F \wedge E$.

- **F e E.** O erro de Sampson mediano fica abaixo de 0,5 px nos três detectores, dentro do limiar de 1,0 px do USAC_MAGSAC. O AKAZE produz o maior número de inliers por par.
- **R e t.** A pose foi recuperada em 55 ou 56 dos 60 pares por detector. Os pares sem pose têm menos de 5 inliers ou, em um par do ORB (`img_0000__img_0001`), nenhum ponto à frente das duas câmeras. A rotação mediana entre quadros consecutivos fica entre 7,7° e 8,0°.
- **Linhas epipolares.** Foram geradas 55 figuras para o AKAZE, 52 para o ORB e 55 para o SIFT; os pares com menos de 8 inliers não têm figura.
- **Nuvem inicial.** Todos os pontos mantidos estão à frente das duas câmeras (fração quiral mediana 1,00). A paralaxe mediana de cerca de 2° confirma a baseline pequena entre quadros consecutivos. Alguns pares com pouquíssimos inliers apresentam paralaxe muito alta (por exemplo, SIFT `img_0069__img_0076`, 164,6° com 8 pontos), indício de pose degenerada nesses pares.

## 8. Limitações

- **Ambiguidade de escala.** Com duas vistas e sem informação métrica, $\mathbf{t}$ e a nuvem são conhecidos apenas a menos de um fator de escala; cada par tem sua própria escala, o que impede juntar as nuvens diretamente.
- **Movimento degenerado.** Em rotação pura ($\mathbf{t} \approx \mathbf{0}$) a matriz essencial degenera, a direção de $\mathbf{t}$ fica indefinida e a triangulação perde sentido (raios praticamente paralelos); nesse caso uma homografia descreveria melhor o par.
- **Cenas planares.** Quando a maior parte dos pontos está num mesmo plano (paredes, piso, mesas, comuns em ambientes internos), a estimação de $F$ fica mal condicionada e várias soluções explicam igualmente os dados.
- **Baseline pequena.** Os pares são de quadros consecutivos, com sobreposição alta (60–80%) e deslocamento pequeno da câmera; a paralaxe resultante é baixa, e a profundidade triangulada é muito sensível a erros de poucos pixels.
- **Intrínsecas aproximadas.** $K$ é estimada por $f = \max(w, h)$ e ponto principal no centro, sem modelo de distorção. Erros em $K$ afetam diretamente $E$, a pose e a triangulação, e podem reduzir o número de inliers comuns a $F$ e $E$.
