import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

valid_workflows = [True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True]

dataset_predicts = []
dataset_predicts.append(np.array([0.9112389609921827,0.9161580335898021,0.9208714856322979,0.9115705574186794,0.9161580335898021,0.9115003444829304,0.901957644768957,0.9161580335898021,0.9205130694119289,0.9389895741503698,0.9297179499858119,0.9252289480405778,0.9296608015557603,0.9297179499858119,0.9252289480405778,0.9015061609038219,0.9297179499858119,0.9253152458098076,0.8684995099329942,0.8783013750034241,0.892230918848231,0.8684995099329942,0.8783013750034241,0.892230918848231,0.8732465951501969,0.8783013750034241,0.892230918848231,0.9206009868812173,0.9298852335842378,0.925171781541742,0.9298852335842378,0.9298852335842378,0.9206009868812173,0.9249675645498301,0.9298852335842378,0.9297581991012227,0.9344181828127278,0.9344181828127278,0.9389895741503698,0.9299744226352182,0.9344181828127278,0.9253152458098076,0.9344181828127278,0.9344181828127278,0.9344999755280183,0.8923827871901433,0.8783013750034241,0.8873701256978588,0.8923827871901433,0.8783013750034241,0.8873701256978588,0.8923827871901433,0.8783013750034241,0.8873701256978588,0.9207288282503268,0.9300562153505088,0.9344999755280183,0.9300562153505088,0.9300562153505088,0.9297865234855225,0.9207288282503268,0.9300562153505088,0.9343295904388647,0.9344181828127278,0.9299291808674937,0.9299291808674937,0.9389895741503698,0.9299291808674937,0.9299291808674937,0.9210832097706565,0.9299291808674937,0.9345005722051355,0.8842844069321327,0.8783013750034241,0.8838435864138653,0.8842844069321327,0.8783013750034241,0.8838435864138653,0.8842844069321327,0.8783013750034241,0.8838435864138653])[valid_workflows])
dataset_predicts.append(np.array([0.08575699678687074,0.04326530216666946,0.04426050178608454,0.04866728419443295,0.04326530216666946,0.06075178723104627,0.11145895279177698,0.04326530216666946,0.049858460272435234,0.06760870646485209,0.15937261603385897,0.0820158340545177,0.0932150087993005,0.15937261603385897,0.09876300720250078,0.05296157746792188,0.15937261603385897,0.13556983855884266,0.13413893022497833,0.047721846631207154,0.0644600891570509,0.13413893022497833,0.047721846631207154,0.0644600891570509,0.1364105557276386,0.047721846631207154,0.0644600891570509,0.07294415813374358,0.08743494421930054,0.07420156586912799,0.14282568253835512,0.08743494421930054,0.05018450045169445,0.07518157581299523,0.08743494421930054,0.10938330467901583,0.15478947795738293,0.1185845487808577,0.15613305997148053,0.04688968782319389,0.1185845487808577,0.11799966644210541,0.049905844598525206,0.1185845487808577,0.13525831785497064,0.10934161435837725,0.047721846631207154,0.044690662989759294,0.10934161435837725,0.047721846631207154,0.044690662989759294,0.08951484423743483,0.047721846631207154,0.044690662989759294,0.10330054416142828,0.12729852225411759,0.1205974647683668,0.12134675936754286,0.12729852225411759,0.09755915123314843,0.10284168744989686,0.12729852225411759,0.1280436288605125,0.13297579612111474,0.11059468283017171,0.154386715863894,0.0848606169627951,0.11059468283017171,0.12733374781421294,0.09918089276260064,0.11059468283017171,0.11799966644210541,0.1700237105201752,0.047721846631207154,0.10202433990807352,0.1700237105201752,0.047721846631207154,0.10202433990807352,0.1700237105201752,0.047721846631207154,0.10202433990807352])[valid_workflows])
dataset_predicts.append(np.array([0.8308471971479493,0.8702013032422607,0.8663850924199925,0.7730298820950922,0.8702013032422607,0.8729294274953885,0.6742703014218663,0.8702013032422607,0.8766948895445003,0.8021031473030832,0.8477023503628276,0.8451594209288137,0.7463972820166038,0.8477023503628276,0.8746889865364934,0.6347912354998401,0.8477023503628276,0.873482843596652,0.45111604872813926,0.6655012534356598,0.8199073088812106,0.45111604872813926,0.6655012534356598,0.8199073088812106,0.45137183656561775,0.6655012534356598,0.8199073088812106,0.847938543075494,0.8800993203521558,0.8724236331048871,0.8099427822419785,0.8800993203521558,0.9011388795811637,0.7155811678036579,0.8800993203521558,0.906626999365368,0.8152090863592245,0.8480751641776749,0.8631091390846976,0.7695339801523117,0.8480751641776749,0.8845421045270647,0.7129478450176125,0.8480751641776749,0.8966484139441864,0.44173547429235616,0.6655012534356598,0.830291657609459,0.44173547429235616,0.6655012534356598,0.830291657609459,0.44173547429235616,0.6655012534356598,0.830291657609459,0.8482975623760254,0.8699973579800886,0.8743475536437769,0.8209040477623306,0.8699973579800886,0.8914873962859009,0.7371303205076103,0.8699973579800886,0.8992265490694602,0.8158085246854578,0.8404953829900896,0.847731010649135,0.7724791236775564,0.8404953829900896,0.8768954685608201,0.7175906213723416,0.8404953829900896,0.8987597415878628,0.43522128756004747,0.6655012534356598,0.8064757446404425,0.43522128756004747,0.6655012534356598,0.8064757446404425,0.43522128756004747,0.6655012534356598,0.8064757446404425])[valid_workflows])
dataset_predicts.append(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.933394474378081,1.0,1.0,0.933394474378081,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9702797202797203,1.0,1.0,0.933394474378081,1.0,1.0,0.933394474378081,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9702797202797203,1.0,1.0,0.933394474378081,1.0,1.0,0.933394474378081,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])[valid_workflows])
dataset_predicts.append(np.array([0.46225497368911167,0.5283254920272424,0.5211419466108832,0.48730497528033423,0.5283254920272424,0.5557556304913174,0.39078384774498753,0.5283254920272424,0.4891474120474787,0.5113674500455042,0.49036065615973,0.47135480820651365,0.4830507146392622,0.49036065615973,0.5215551172185618,0.3074073516393949,0.49036065615973,0.49512483161942644,0.5133682076028503,0.5178906854673981,0.41899014447091376,0.5133682076028503,0.5178906854673981,0.41899014447091376,0.48082709432519316,0.5178906854673981,0.43422472713970695,0.43331285933761265,0.49178060127309653,0.49178060127309653,0.4707026202679171,0.49178060127309653,0.5102419565408343,0.4208543537186898,0.49178060127309653,0.5043995383814743,0.4868857208041334,0.5070448123145963,0.5070448123145963,0.5222394776341354,0.5070448123145963,0.4898892334775088,0.46403418349711123,0.5070448123145963,0.5281299117631533,0.35251515625177454,0.5178906854673981,0.4751603996543268,0.35251515625177454,0.5178906854673981,0.4751603996543268,0.33643328490674534,0.5178906854673981,0.4751603996543268,0.5005791525223872,0.4834235736852997,0.4834235736852997,0.5363520407351359,0.4834235736852997,0.4834235736852997,0.36151048320978263,0.4834235736852997,0.5102419565408343,0.5406705433087483,0.5070448123145963,0.5070448123145963,0.5589112022143063,0.5070448123145963,0.5070448123145963,0.5163285880024746,0.5070448123145963,0.5070448123145963,0.49948735475051265,0.5178906854673981,0.49238102129177297,0.49948735475051265,0.5178906854673981,0.49238102129177297,0.49948735475051265,0.5178906854673981,0.49238102129177297])[valid_workflows])
dataset_predicts.append(np.array([0.3636309803157944,0.3636309803157944,0.3636309803157944,0.2960393153187998,0.3636309803157944,0.3185943307731306,0.2811451305386489,0.3636309803157944,0.3519679382239712,0.3579157484296107,0.3968111345334126,0.3968111345334126,0.37469447430793057,0.3968111345334126,0.3636309803157944,0.36787383194055423,0.3968111345334126,0.3958428338849761,0.27257940436894423,0.27853480790093343,0.2606431571254998,0.2495999911106138,0.27853480790093343,0.2606431571254998,0.23674158307066054,0.27853480790093343,0.2606431571254998,0.3416672317318653,0.2869588302149304,0.2991565322420068,0.2869588302149304,0.2869588302149304,0.29265168525774027,0.21446615469205813,0.2869588302149304,0.2908046653080386,0.3416672317318653,0.3416672317318653,0.3636309803157944,0.3579157484296107,0.3416672317318653,0.35301473790373306,0.3310628689130243,0.3416672317318653,0.36205887627761435,0.263380035746166,0.27853480790093343,0.24898546010414355,0.263380035746166,0.27853480790093343,0.24898546010414355,0.263380035746166,0.27853480790093343,0.24898546010414355,0.2960393153187998,0.3175654544655311,0.2991565322420068,0.3175654544655311,0.3175654544655311,0.323258309508341,0.27193753805246557,0.3175654544655311,0.3243549520985869,0.3234654265941658,0.36795884840624893,0.3370773082441266,0.3641969049600195,0.36795884840624893,0.3454291751780948,0.3234654265941658,0.36795884840624893,0.3725853111077436,0.19466719831756582,0.27853480790093343,0.15192124568036444,0.19466719831756582,0.27853480790093343,0.15192124568036444,0.19466719831756582,0.27853480790093343,0.15192124568036444])[valid_workflows])
dataset_predicts.append(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])[valid_workflows])
dataset_predicts.append(np.array([0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,-0.025575447570332477,-0.01470588235294118,-0.01470588235294118,-0.025575447570332477,-0.01470588235294118,-0.25,-0.025575447570332477,-0.01470588235294118,-0.025575447570332477,-0.010869565217391297,-0.025575447570332477,-0.025575447570332477,-0.010869565217391297,-0.025575447570332477,-0.025575447570332477,-0.010869565217391297,-0.025575447570332477,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,-0.025575447570332477,-0.25,-0.25,-0.025575447570332477,-0.25,-0.25,-0.025575447570332477,-0.25,-0.25,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,0.05618964003511853,0.05618964003511853,-0.025575447570332477,-0.025575447570332477,-0.25,-0.25,-0.025575447570332477,-0.25,-0.25,-0.025575447570332477,-0.25,-0.25,-0.25,-0.025575447570332477,-0.025575447570332477,-0.25,-0.025575447570332477,-0.025575447570332477,-0.25,-0.025575447570332477,-0.025575447570332477])[valid_workflows])
dataset_predicts.append(np.array([0.5064528360306368,0.5151625282918547,0.509853393917203,0.50190646015848,0.5151625282918547,0.5027795148900033,0.4885870682602599,0.5151625282918547,0.4994659629430221,0.5373830203771512,0.5309836341663757,0.5325938371949692,0.5265712484272485,0.5309836341663757,0.5401475786117396,0.5140494972333132,0.5309836341663757,0.5175219137311841,0.4347390632499787,0.4427735076056722,0.46690241870119187,0.4347390632499787,0.4427735076056722,0.46690241870119187,0.4380526145108504,0.4427735076056722,0.46321178329896007,0.5225122851580324,0.5207764522687693,0.5156842631988353,0.5159778522268069,0.5207764522687693,0.5094896846977623,0.5008172165253831,0.5207764522687693,0.49300752549602966,0.5402484382091758,0.5353812273988057,0.5409214755042349,0.5438560524441619,0.5353812273988057,0.5279676884953751,0.5271272327640979,0.5353812273988057,0.5258158364327469,0.41839130263971114,0.4427735076056722,0.44731590788077236,0.41839130263971114,0.4427735076056722,0.44731590788077236,0.4194875754226727,0.4427735076056722,0.4462193309783735,0.514704529862226,0.5235271505804909,0.5234024841516847,0.5195279890849043,0.5235271505804909,0.5189240136755088,0.5083786224603394,0.5235271505804909,0.5080370792587222,0.5422540293506526,0.5350361877520872,0.5359615792724373,0.5366799404574881,0.5350361877520872,0.5316213124628582,0.5410134573698099,0.5350361877520872,0.5290368075507964,0.414976457023204,0.4427735076056722,0.43527215154920285,0.414976457023204,0.4427735076056722,0.43527215154920285,0.414976457023204,0.4427735076056722,0.43527215154920285])[valid_workflows])
dataset_predicts.append(np.array([0.2075267830181099,0.24939173093246877,0.26104987930390167,0.16286838157072117,0.24939173093246877,0.2339520046890491,0.157251698576432,0.24939173093246877,0.2132894190530508,0.329612351865296,0.32898103873398293,0.31646373545911577,0.3438613521172298,0.32898103873398293,0.3120270338427723,0.27997675583272474,0.32898103873398293,0.3492110175864346,0.19887571827634917,0.3115489299956645,0.23583769748910324,0.18751217120735655,0.3115489299956645,0.23583769748910324,0.22894887631333413,0.3115489299956645,0.23583769748910324,0.21531288478283878,0.24275694938825515,0.25160097978733303,0.2232643925167973,0.24275694938825515,0.25885394380667953,0.17217577016297017,0.24275694938825515,0.2554782974419207,0.3054303728489455,0.2888675569670208,0.34324744779052363,0.32893312449284473,0.2888675569670208,0.3156807096590023,0.2916358457729621,0.2888675569670208,0.30461238197470414,0.2224304400395216,0.3115489299956645,0.23569689077533673,0.2224304400395216,0.3115489299956645,0.23569689077533673,0.2348290965375304,0.3115489299956645,0.23569689077533673,0.20908691747395375,0.23638621058864365,0.2556276617698604,0.20825967963828085,0.23638621058864365,0.23776383163798775,0.2098369336490505,0.23638621058864365,0.246689664610537,0.35981787250485575,0.35981787250485575,0.37083960816026595,0.33714209610979173,0.35981787250485575,0.3477404407121926,0.3234952766679263,0.35981787250485575,0.35016840483752787,0.31037488193005747,0.3115489299956645,0.28486000898649405,0.31037488193005747,0.3115489299956645,0.28486000898649405,0.31037488193005747,0.3115489299956645,0.28486000898649405])[valid_workflows])
dataset_predicts.append(np.array([0.1811534334431686,0.17926336320234232,0.16697104908148955,0.18837901959878955,0.17926336320234232,0.1561367098651306,0.2022617515474079,0.17926336320234232,0.17045600431531355,0.19466423777363606,0.19849552667598683,0.20802501504340082,0.2152439280516718,0.19849552667598683,0.1850633997528097,0.20606673412283988,0.19849552667598683,0.17433259801397671,0.14174781373428774,0.1303883191379006,0.14718377584049736,0.14174781373428774,0.1303883191379006,0.14718377584049736,0.13331301914789917,0.1303883191379006,0.14718377584049736,0.17225861426455086,0.1668313267527136,0.17322899935763394,0.20718501088203733,0.1668313267527136,0.1793895813950275,0.1872633453205186,0.1668313267527136,0.19710018647848385,0.2119549688646422,0.21080649418978237,0.19656424681328588,0.2054023386046931,0.21080649418978237,0.20309763163079944,0.18027873699987734,0.21080649418978237,0.17794077781423556,0.18618620713426173,0.1303883191379006,0.1258536808178296,0.18618620713426173,0.1303883191379006,0.1258536808178296,0.189127510446223,0.1303883191379006,0.1258536808178296,0.1887697707036908,0.18404950479555848,0.17148946846495966,0.2115141870246764,0.18404950479555848,0.1880120196701895,0.20867916358126062,0.18404950479555848,0.19616499068638632,0.19987648671129296,0.19689625954331252,0.1903708306972298,0.19773837489325202,0.19689625954331252,0.1850590068674834,0.18358027841750518,0.19689625954331252,0.1875738103173484,0.11086568209645517,0.1303883191379006,0.14788272572756858,0.11086568209645517,0.1303883191379006,0.14788272572756858,0.11086568209645517,0.1303883191379006,0.14788272572756858])[valid_workflows])
dataset_predicts.append(np.array([0.09876407844707694,0.12276610892743542,0.17118321990493487,0.05527884181740178,0.12276610892743542,0.16064125963040612,0.026074490341036427,0.12276610892743542,0.146691826423398,0.14196200373964435,0.14938350571834735,0.1282316780547146,0.10630203174138239,0.14938350571834735,0.1712902482133251,0.14196200373964435,0.14938350571834735,0.1554628198907077,0.06802401691782059,0.09321786030196794,0.18625048766937086,0.06802401691782059,0.09321786030196794,0.18625048766937086,0.06802401691782059,0.09321786030196794,0.18625048766937086,0.11143854478533635,0.12276610892743542,0.12276610892743542,0.10167769015514952,0.12276610892743542,0.11143854478533635,0.05134382420861078,0.12276610892743542,0.16805402503242492,0.15044701315402179,0.1578685151327248,0.1578685151327248,0.15304923802429107,0.1578685151327248,0.14196200373964435,0.14938350571834735,0.1578685151327248,0.1345634055792958,0.14846546277971753,0.09321786030196794,0.12484921369119834,0.14846546277971753,0.09321786030196794,0.12484921369119834,0.14846546277971753,0.09321786030196794,0.12484921369119834,0.13563027621506316,0.1386497165021614,0.13016470708778397,0.13081099910662944,0.1386497165021614,0.11800034524757128,0.08874458874458871,0.1386497165021614,0.09671094707293154,0.16668763011814416,0.16027668805232265,0.14196200373964435,0.15304923802429107,0.16027668805232265,0.14938350571834735,0.10420761770239281,0.16027668805232265,0.10566898573394048,0.1340628668599009,0.09321786030196794,0.16471766490525216,0.1340628668599009,0.09321786030196794,0.16471766490525216,0.1340628668599009,0.09321786030196794,0.16471766490525216])[valid_workflows])
dataset_predicts.append(np.array([0.3494569202695953,0.3190196843242197,0.3304219297353132,0.30928139341448335,0.3190196843242197,0.2905661407066993,0.28694384965798275,0.3190196843242197,0.32131191254637137,0.3173011978965622,0.27702755344538954,0.2765905663784416,0.28962779524423987,0.27702755344538954,0.25631012230001676,0.26718079082147966,0.27702755344538954,0.28161176546209143,0.28688592156732823,0.3030674005737789,0.24426249617810972,0.28688592156732823,0.3030674005737789,0.24426249617810972,0.28688592156732823,0.3030674005737789,0.24426249617810972,0.35272094142275645,0.30363785067370574,0.31676705691300305,0.3303829543049802,0.30363785067370574,0.32656674053722445,0.3512812596971265,0.30363785067370574,0.26762507541900193,0.282368834875525,0.3024889142286501,0.29927975180254496,0.34161795988574184,0.3024889142286501,0.2665087367590244,0.3071937500313375,0.3024889142286501,0.25992315324835974,0.24733433652027326,0.3030674005737789,0.24870800663009435,0.24733433652027326,0.3030674005737789,0.24870800663009435,0.24733433652027326,0.3030674005737789,0.24870800663009435,0.30013610658547435,0.2838544856125399,0.27177474471641144,0.2898198597820044,0.2838544856125399,0.28429014707670747,0.29691567437677,0.2838544856125399,0.2871069798090111,0.2973698792218603,0.29216345957379697,0.2978215992864317,0.26634173703027886,0.29216345957379697,0.2981125903901406,0.29452501618220545,0.29216345957379697,0.28098880915387925,0.2344658773345258,0.3030674005737789,0.2512541323904295,0.2344658773345258,0.3030674005737789,0.2512541323904295,0.2344658773345258,0.3030674005737789,0.2512541323904295])[valid_workflows])
dataset_predicts.append(np.array([0.737399987538683,0.7422736211309255,0.739390515116852,0.7222790221234463,0.7422736211309255,0.7335742438133205,0.71251276282323,0.7422736211309255,0.7075302203245633,0.7536875102243255,0.740115843472195,0.7396614450182175,0.7352168888586155,0.740115843472195,0.7400871051579441,0.7219483004933815,0.740115843472195,0.710445787176174,0.6429573429476687,0.6436150065190611,0.6439514430284964,0.642412012248202,0.6436150065190611,0.6439514430284964,0.6409395114612764,0.6436150065190611,0.6451955568598897,0.7582685262953694,0.7520056465499565,0.7524260301885827,0.7440701038911539,0.7520056465499565,0.7496486120598114,0.7283334820336009,0.7520056465499565,0.7441475088430258,0.7522282148454998,0.7557568671084957,0.747656066025804,0.7498856938748717,0.7557568671084957,0.7483529198779832,0.746964834183683,0.7557568671084957,0.7341371986199314,0.6211750476096285,0.6436150065190611,0.6303056228012409,0.6211750476096285,0.6436150065190611,0.6303056228012409,0.6211750476096285,0.6436150065190611,0.6303056228012409,0.7614461149442834,0.7632478697238821,0.7603662801198775,0.7588835869966879,0.7632478697238821,0.75975441616272,0.7470492317734225,0.7632478697238821,0.7468064580647007,0.7577238931524566,0.7559544274285552,0.7573449211185064,0.7526156301061957,0.7559544274285552,0.7515354527346949,0.7477868940594206,0.7559544274285552,0.7405220235141652,0.6288760021147296,0.6436150065190611,0.6469970776000624,0.6288760021147296,0.6436150065190611,0.6469970776000624,0.6288760021147296,0.6436150065190611,0.6469970776000624])[valid_workflows])
dataset_predicts.append(np.array([0.6893310277494025,0.7004542912378869,0.6817541381745609,0.6806288915562226,0.7004542912378869,0.680659246234332,0.611804042907791,0.7004542912378869,0.6649018960131194,0.682878491566284,0.6768674140213491,0.6620500473682385,0.6845981229240713,0.6768674140213491,0.6786187857933348,0.688713045947106,0.6768674140213491,0.6756530440927935,0.5518555513230707,0.5575379760393665,0.5937430313764946,0.5518555513230707,0.5575379760393665,0.5901878095660291,0.5518555513230707,0.5575379760393665,0.5901878095660291,0.718045357881452,0.693384293433687,0.6890770014243268,0.7006426192422246,0.693384293433687,0.690026341169567,0.673649891324381,0.693384293433687,0.6794634640347044,0.7101217725807414,0.6912885131615369,0.6902143800140442,0.71120329412849,0.6912885131615369,0.6933227209716142,0.6863715970764839,0.6912885131615369,0.6737191287166946,0.5587592327691543,0.5575379760393665,0.5884597224687973,0.5587592327691543,0.5575379760393665,0.5884597224687973,0.5587592327691543,0.5575379760393665,0.5848285908626477,0.7013787823421644,0.7021435161796598,0.7077148071705877,0.6920312862466799,0.7021435161796598,0.6944060563928127,0.6908292178618208,0.7021435161796598,0.679362383383767,0.7054013043541083,0.698359859420361,0.6992261463422379,0.7023345658959496,0.698359859420361,0.6935143466074261,0.6927363248221209,0.698359859420361,0.6843827525993331,0.5879521126780503,0.5575379760393665,0.601238391751791,0.5879521126780503,0.5575379760393665,0.601238391751791,0.5879521126780503,0.5575379760393665,0.601238391751791])[valid_workflows])
dataset_predicts.append(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9952234684100772,1.0,1.0,0.9952234684100772,1.0,1.0,0.9952234684100772,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.992050050452265,1.0,1.0,0.992050050452265,1.0,1.0,0.992050050452265,1.0,1.0])[valid_workflows])
dataset_predicts.append(np.array([0.8433900724620595,0.8416298479350163,0.8361086234877906,0.8372675150781655,0.8416298479350163,0.8429168312372443,0.8269188288529384,0.8416298479350163,0.8290991604204517,0.8381232360919695,0.8381232360919695,0.8428803786010943,0.8303023143890299,0.8381232360919695,0.8285296701065895,0.8244028184529996,0.8381232360919695,0.8250436332680601,0.7247726673746746,0.772269142641634,0.7705217643379972,0.7247726673746746,0.772269142641634,0.7705217643379972,0.7247726673746746,0.772269142641634,0.7705217643379972,0.8411724753442484,0.8377753592832864,0.8341510378006899,0.8427887235988303,0.8377753592832864,0.8366205477717554,0.8455140939254616,0.8377753592832864,0.8338637189762902,0.835305849312221,0.8389287262468272,0.8377739147352963,0.831718074941344,0.8389287262468272,0.8375677325768325,0.8389574893370971,0.8389287262468272,0.8382605705310482,0.7099108604160729,0.772269142641634,0.784671754306661,0.7099108604160729,0.772269142641634,0.784671754306661,0.7099108604160729,0.772269142641634,0.784671754306661,0.8444994517472977,0.8314408441227208,0.8391633793902922,0.8426603502609779,0.8314408441227208,0.8367905793348642,0.8430880855790739,0.8314408441227208,0.8341510378006899,0.8391633793902922,0.8391633793902922,0.8427862563248985,0.8357044229233371,0.8391633793902922,0.8377739147352963,0.8427422911140561,0.8391633793902922,0.8377739147352963,0.6966482855306141,0.772269142641634,0.7792151687297243,0.6966482855306141,0.772269142641634,0.7792151687297243,0.6966482855306141,0.772269142641634,0.7792151687297243])[valid_workflows])
dataset_predicts.append(np.array([0.7675095249565673,0.7726915274349575,0.7571146181264118,0.7493589347086843,0.7726915274349575,0.7544163348978414,0.7493261153829547,0.7726915274349575,0.745420336370724,0.766078581467012,0.7681319263725472,0.7637251057681356,0.7533335990496511,0.7681319263725472,0.7644117077645434,0.7487888368534235,0.7681319263725472,0.7466455524811433,0.6059990344021171,0.6217037476208174,0.6485876796984488,0.6059990344021171,0.6217037476208174,0.6485876796984488,0.6067799363971149,0.6217037476208174,0.6485876796984488,0.7771032487348725,0.7763794852282784,0.7742685541216441,0.7728550959778285,0.7763794852282784,0.7650377826601529,0.7540540865504459,0.7763794852282784,0.7480119180025075,0.7763421506706775,0.7763834814896639,0.7737704163763577,0.775114693002389,0.7763834814896639,0.7631928007467643,0.7674824256182597,0.7763834814896639,0.7605719830333884,0.5895397553307553,0.6217037476208174,0.6512291264954277,0.5895397553307553,0.6217037476208174,0.6512291264954277,0.5895397553307553,0.6217037476208174,0.6512291264954277,0.7814031823056296,0.784312524196203,0.7813046167432924,0.7747050337878618,0.784312524196203,0.7776235263518456,0.7606220322507373,0.784312524196203,0.767206682501558,0.7812490323009561,0.7817710997180014,0.7808431355682189,0.776038316866029,0.7817710997180014,0.7750643130469967,0.7670655291792794,0.7817710997180014,0.7744874081465773,0.5724869089889577,0.6217037476208174,0.6526767686548498,0.5724869089889577,0.6217037476208174,0.6526767686548498,0.5724869089889577,0.6217037476208174,0.6526767686548498])[valid_workflows])
dataset_predicts.append(np.array([0.7498745999099571,0.7498745999099571,0.7399093333414212,0.7733678325123581,0.7498745999099571,0.7399093333414212,0.7939507539620323,0.7498745999099571,0.7118682700648263,0.7828682489182328,0.7729029823496969,0.7626125218811171,0.7830790899474154,0.7729029823496969,0.7471781826367602,0.7829622139332175,0.7729029823496969,0.7274734119514377,0.6889082100011519,0.642069174674101,0.6870050664307311,0.6889082100011519,0.642069174674101,0.6870050664307311,0.6889082100011519,0.642069174674101,0.6870050664307311,0.7675173583851991,0.7512929750781372,0.7443780062953839,0.7575965765114712,0.7512929750781372,0.7443780062953839,0.7884032742718958,0.7512929750781372,0.7575965765114712,0.7985134291917722,0.7678087727994662,0.7678087727994662,0.7820337201607197,0.7678087727994662,0.7562275783449267,0.7959211890729891,0.7678087727994662,0.7596375278556515,0.6659316846927705,0.642069174674101,0.6824469999431618,0.6659316846927705,0.642069174674101,0.6824469999431618,0.6659316846927705,0.642069174674101,0.6824469999431618,0.7606468742972538,0.7506816077287178,0.7506816077287178,0.7675618430800071,0.7506816077287178,0.7443780062953839,0.782951697629556,0.7506816077287178,0.766727314322494,0.7756487766245295,0.7756487766245295,0.7777740393680022,0.7756487766245295,0.7756487766245295,0.7562275783449267,0.8006717082004224,0.7756487766245295,0.7719235652408434,0.6665420691511736,0.642069174674101,0.7071973390230664,0.6665420691511736,0.642069174674101,0.7071973390230664,0.6665420691511736,0.642069174674101,0.7071973390230664])[valid_workflows])
dataset_predicts.append(np.array([-0.0011682242990654346,-0.0018744142455482393,-0.0021321961620469065,-0.0018744142455482393,-0.0018744142455482393,-0.0021321961620469065,0.09078236900517844,-0.0018744142455482393,-0.003531449893390215,-0.0011682242990654346,-0.0018744142455482393,-0.0021321961620469065,-0.0018744142455482393,-0.0018744142455482393,-0.0021321961620469065,0.04485992589354401,-0.0018744142455482393,-0.003531449893390215,0.04065767719125066,0.0324009465980514,0.010243186647594965,0.04065767719125066,0.0324009465980514,0.010243186647594965,0.04065767719125066,0.0324009465980514,0.010243186647594965,-0.0011682242990654346,-0.0015600624024960652,-0.0018744142455482393,-0.0011682242990654346,-0.0015600624024960652,-0.0018744142455482393,-0.0006662225183211579,-0.0015600624024960652,-0.0032736679768915478,-0.0011682242990654346,-0.0018744142455482393,-0.0018744142455482393,-0.0006662225183211579,-0.0018744142455482393,-0.0018744142455482393,-0.0006662225183211579,-0.0018744142455482393,-0.0025740411112198935,0.040787962563606756,0.0324009465980514,0.026585406911549342,0.040787962563606756,0.0324009465980514,0.026585406911549342,0.040787962563606756,0.0324009465980514,0.026585406911549342,-0.001365849383992812,-0.0018744142455482393,-0.0018744142455482393,-0.001365849383992812,-0.0018744142455482393,-0.0018744142455482393,0.0,-0.0018744142455482393,-0.0025740411112198935,-0.001365849383992812,-0.0018744142455482393,-0.0018744142455482393,-0.001365849383992812,-0.0018744142455482393,-0.0018744142455482393,0.0,-0.0018744142455482393,-0.0025740411112198935,0.03844505745964508,0.0324009465980514,-0.017107861347457787,0.03844505745964508,0.0324009465980514,-0.017107861347457787,0.03844505745964508,0.0324009465980514,-0.017107861347457787])[valid_workflows])
dataset_predicts.append(np.array([0.8433900724620595,0.8416298479350163,0.8361086234877906,0.8372675150781655,0.8416298479350163,0.8429168312372443,0.8269188288529384,0.8416298479350163,0.8290991604204517,0.8381232360919695,0.8381232360919695,0.8428803786010943,0.8303023143890299,0.8381232360919695,0.8285296701065895,0.8244028184529996,0.8381232360919695,0.8250436332680601,0.7247726673746746,0.772269142641634,0.7705217643379972,0.7247726673746746,0.772269142641634,0.7705217643379972,0.7247726673746746,0.772269142641634,0.7705217643379972,0.8411724753442484,0.8377753592832864,0.8341510378006899,0.8427887235988303,0.8377753592832864,0.8366205477717554,0.8455140939254616,0.8377753592832864,0.8338637189762902,0.835305849312221,0.8389287262468272,0.8377739147352963,0.831718074941344,0.8389287262468272,0.8375677325768325,0.8389574893370971,0.8389287262468272,0.8382605705310482,0.7099108604160729,0.772269142641634,0.784671754306661,0.7099108604160729,0.772269142641634,0.784671754306661,0.7099108604160729,0.772269142641634,0.784671754306661,0.8444994517472977,0.8314408441227208,0.8391633793902922,0.8426603502609779,0.8314408441227208,0.8367905793348642,0.8430880855790739,0.8314408441227208,0.8341510378006899,0.8391633793902922,0.8391633793902922,0.8427862563248985,0.8357044229233371,0.8391633793902922,0.8377739147352963,0.8427422911140561,0.8391633793902922,0.8377739147352963,0.6966482855306141,0.772269142641634,0.7792151687297243,0.6966482855306141,0.772269142641634,0.7792151687297243,0.6966482855306141,0.772269142641634,0.7792151687297243])[valid_workflows])
dataset_predicts.append(np.array([0.4723526186754177,0.4711633646151091,0.510373435287384,0.4226281828300256,0.4711633646151091,0.5612336308165001,0.33524419480882495,0.4711633646151091,0.5005144306080068,0.31678707738017065,0.4628459345853302,0.5300311288643362,0.34079891125704453,0.4628459345853302,0.5257728700808101,0.34768062159801183,0.4628459345853302,0.5743364710757453,0.12996911101670092,0.3240621603295705,0.4883803321115539,0.12996911101670092,0.3240621603295705,0.4883803321115539,0.12996911101670092,0.3240621603295705,0.4883803321115539,0.4325151038941141,0.44005736522912337,0.5173294237575259,0.4063627732484928,0.44005736522912337,0.5345820094827592,0.34493636748189105,0.44005736522912337,0.5576523244763327,0.36298567755162525,0.40915994026078695,0.5481467828981447,0.3040444077897679,0.40915994026078695,0.5641405288744162,0.31860762811584437,0.40915994026078695,0.5821717038346192,0.1353289712762935,0.3240621603295705,0.5627656061782672,0.1353289712762935,0.3240621603295705,0.5627656061782672,0.1353289712762935,0.3240621603295705,0.5627656061782672,0.3973027754841737,0.3973027754841737,0.5206390664924319,0.3481893892024073,0.3973027754841737,0.6022370291924402,0.2884545865465338,0.3973027754841737,0.5399581964856113,0.3991071137556622,0.42992447289628105,0.4969380894928448,0.31004406662610784,0.42992447289628105,0.5354626796567792,0.22096386447299232,0.42992447289628105,0.5449682212349672,0.17094914106308065,0.3240621603295705,0.45672849864902004,0.17094914106308065,0.3240621603295705,0.45672849864902004,0.17094914106308065,0.3240621603295705,0.45672849864902004])[valid_workflows])

bagging_predicts = []
labels = []
#for i in range(0,63):
#    bagging_predicts.append([])
#    labels.append("Teste")

#for dataset_predict in dataset_predicts:
#    for i in range(0,63):
#        bagging_predicts[i].append(dataset_predict[i])

df = pd.DataFrame({
    'A' : dataset_predicts[0],
    'B' : dataset_predicts[1],
    'C' : dataset_predicts[2],
    'D' : dataset_predicts[3],
    'E' : dataset_predicts[4],
    'F' : dataset_predicts[5],
    'G' : dataset_predicts[6],
    'H' : dataset_predicts[7],
    'I' : dataset_predicts[8],
    'J' : dataset_predicts[9],
    'K' : dataset_predicts[10],
    'L' : dataset_predicts[11],
    'M' : dataset_predicts[12],
    'N' : dataset_predicts[13],
    'O' : dataset_predicts[14],
    'P' : dataset_predicts[15],
    'Q' : dataset_predicts[16],
    'R' : dataset_predicts[17],
    'S' : dataset_predicts[18],
    'T' : dataset_predicts[19],
    'U' : dataset_predicts[20],
    'V' : dataset_predicts[21]
})
df = df.reindex(df.sum().sort_values().index, axis=1)
boxplot = df.boxplot(grid=False, rot=90)

# set x-axis label
plt.xlabel("Datasets", size=18)
# set y-axis label
plt.ylabel("Kappa values", size=18)
#print(bagging_predicts)

#df.boxplot()
plt.show()

# LAYOUT PARA O GRAFICO
#top=0.995,
#bottom=0.345,
#left=0.07,
#right=0.99,
#hspace=0.2,
#wspace=0.2