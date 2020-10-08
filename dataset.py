import os
import librosa
import vocabulary
from multiprocessing import Pool
import audio


# 화자 이름 얻기
"""
speakers = []
for speaker in os.listdir("E:\\train-clean-100\\LibriSpeech\\train-clean-100"):
    speakers.append("\'libri_" + speaker + "\'")
print("[" + ", ".join(speakers) + "]")

speakers = []
for speaker in os.listdir("E:\\zeroth_korean\\train_data_01\\003"):
    speakers.append("\'zeroth_" + speaker + "\'")
print("[" + ", ".join(speakers) + "]")

speakers = []
for speaker in os.listdir("E:\\NIKL"):
    if os.path.isdir("E:\\NIKL\\" + speaker):
        speakers.append("\'nikl_" + speaker + "\'")
print("[" + ", ".join(speakers) + "]")

exit(0)
"""

LIBRI_SPEAKERS = ['libri_103', 'libri_1034', 'libri_1040', 'libri_1069', 'libri_1081', 'libri_1088', 'libri_1098', 'libri_1116', 'libri_118', 'libri_1183', 'libri_1235', 'libri_1246', 'libri_125', 'libri_1263', 'libri_1334', 'libri_1355', 'libri_1363', 'libri_1447', 'libri_1455', 'libri_150', 'libri_1502', 'libri_1553', 'libri_1578', 'libri_1594', 'libri_1624', 'libri_163', 'libri_1723', 'libri_1737', 'libri_1743', 'libri_1841', 'libri_1867', 'libri_1898', 'libri_19', 'libri_1926', 'libri_196', 'libri_1963', 'libri_1970', 'libri_198', 'libri_1992', 'libri_200', 'libri_2002', 'libri_2007', 'libri_201', 'libri_2092', 'libri_211', 'libri_2136', 'libri_2159', 'libri_2182', 'libri_2196', 'libri_226', 'libri_2289', 'libri_229', 'libri_233', 'libri_2384', 'libri_2391', 'libri_2416', 'libri_2436', 'libri_248', 'libri_250', 'libri_2514', 'libri_2518', 'libri_254', 'libri_26', 'libri_2691', 'libri_27', 'libri_2764', 'libri_2817', 'libri_2836', 'libri_2843', 'libri_289', 'libri_2893', 'libri_2910', 'libri_2911', 'libri_2952', 'libri_298', 'libri_2989', 'libri_302', 'libri_307', 'libri_311', 'libri_3112', 'libri_3168', 'libri_32', 'libri_3214', 'libri_322', 'libri_3235', 'libri_3240', 'libri_3242', 'libri_3259', 'libri_328', 'libri_332', 'libri_3374', 'libri_3436', 'libri_3440', 'libri_3486', 'libri_3526', 'libri_3607', 'libri_3664', 'libri_3699', 'libri_3723', 'libri_374', 'libri_3807', 'libri_3830', 'libri_3857', 'libri_3879', 'libri_39', 'libri_3947', 'libri_3982', 'libri_3983', 'libri_40', 'libri_4014', 'libri_4018', 'libri_403', 'libri_405', 'libri_4051', 'libri_4088', 'libri_412', 'libri_4137', 'libri_4160', 'libri_4195', 'libri_4214', 'libri_426', 'libri_4267', 'libri_4297', 'libri_4340', 'libri_4362', 'libri_4397', 'libri_4406', 'libri_441', 'libri_4441', 'libri_445', 'libri_446', 'libri_4481', 'libri_458', 'libri_460', 'libri_4640', 'libri_4680', 'libri_4788', 'libri_481', 'libri_4813', 'libri_4830', 'libri_4853', 'libri_4859', 'libri_4898', 'libri_5022', 'libri_5049', 'libri_5104', 'libri_5163', 'libri_5192', 'libri_5322', 'libri_5339', 'libri_5390', 'libri_5393', 'libri_5456', 'libri_5463', 'libri_5514', 'libri_5561', 'libri_5652', 'libri_5678', 'libri_5688', 'libri_5703', 'libri_5750', 'libri_5778', 'libri_5789', 'libri_5808', 'libri_5867', 'libri_587', 'libri_60', 'libri_6000', 'libri_6019', 'libri_6064', 'libri_6078', 'libri_6081', 'libri_6147', 'libri_6181', 'libri_6209', 'libri_625', 'libri_6272', 'libri_6367', 'libri_6385', 'libri_6415', 'libri_6437', 'libri_6454', 'libri_6476', 'libri_6529', 'libri_6531', 'libri_6563', 'libri_669', 'libri_6818', 'libri_6836', 'libri_6848', 'libri_6880', 'libri_6925', 'libri_696', 'libri_7059', 'libri_7067', 'libri_7078', 'libri_7113', 'libri_7148', 'libri_7178', 'libri_7190', 'libri_7226', 'libri_7264', 'libri_7278', 'libri_730', 'libri_7302', 'libri_7312', 'libri_7367', 'libri_7402', 'libri_7447', 'libri_7505', 'libri_7511', 'libri_7517', 'libri_7635', 'libri_7780', 'libri_7794', 'libri_78', 'libri_7800', 'libri_7859', 'libri_8014', 'libri_8051', 'libri_8063', 'libri_8088', 'libri_8095', 'libri_8098', 'libri_8108', 'libri_8123', 'libri_8226', 'libri_8238', 'libri_83', 'libri_831', 'libri_8312', 'libri_8324', 'libri_839', 'libri_8419', 'libri_8425', 'libri_8465', 'libri_8468', 'libri_8580', 'libri_8609', 'libri_8629', 'libri_8630', 'libri_87', 'libri_8747', 'libri_8770', 'libri_8797', 'libri_8838', 'libri_887', 'libri_89', 'libri_8975', 'libri_909', 'libri_911']
ZEROTH_SPEAKERS = ['zeroth_106', 'zeroth_107', 'zeroth_108', 'zeroth_109', 'zeroth_110', 'zeroth_111', 'zeroth_113', 'zeroth_114', 'zeroth_115', 'zeroth_116', 'zeroth_117', 'zeroth_119', 'zeroth_120', 'zeroth_122', 'zeroth_123', 'zeroth_124', 'zeroth_125', 'zeroth_127', 'zeroth_128', 'zeroth_129', 'zeroth_130', 'zeroth_131', 'zeroth_133', 'zeroth_134', 'zeroth_135', 'zeroth_136', 'zeroth_138', 'zeroth_139', 'zeroth_140', 'zeroth_141', 'zeroth_142', 'zeroth_143', 'zeroth_144', 'zeroth_145', 'zeroth_146', 'zeroth_148', 'zeroth_150', 'zeroth_151', 'zeroth_152', 'zeroth_153', 'zeroth_154', 'zeroth_155', 'zeroth_156', 'zeroth_157', 'zeroth_158', 'zeroth_159', 'zeroth_160', 'zeroth_161', 'zeroth_162', 'zeroth_163', 'zeroth_164', 'zeroth_165', 'zeroth_166', 'zeroth_167', 'zeroth_168', 'zeroth_169', 'zeroth_170', 'zeroth_171', 'zeroth_172', 'zeroth_173', 'zeroth_174', 'zeroth_175', 'zeroth_176', 'zeroth_177', 'zeroth_178', 'zeroth_179', 'zeroth_180', 'zeroth_181', 'zeroth_182', 'zeroth_183', 'zeroth_184', 'zeroth_185', 'zeroth_186', 'zeroth_187', 'zeroth_188', 'zeroth_189', 'zeroth_190', 'zeroth_191', 'zeroth_192', 'zeroth_193', 'zeroth_194', 'zeroth_195', 'zeroth_196', 'zeroth_197', 'zeroth_198', 'zeroth_199', 'zeroth_200', 'zeroth_201', 'zeroth_202', 'zeroth_203', 'zeroth_204', 'zeroth_205', 'zeroth_206', 'zeroth_207', 'zeroth_208', 'zeroth_209', 'zeroth_210', 'zeroth_211', 'zeroth_212', 'zeroth_213', 'zeroth_214', 'zeroth_215', 'zeroth_216', 'zeroth_217', 'zeroth_218']
NIKL_SPEAKERS = ['nikl_fv01', 'nikl_fv02', 'nikl_fv03', 'nikl_fv04', 'nikl_fv05', 'nikl_fv06', 'nikl_fv07', 'nikl_fv08', 'nikl_fv09', 'nikl_fv10', 'nikl_fv11', 'nikl_fv12', 'nikl_fv13', 'nikl_fv14', 'nikl_fv15', 'nikl_fv16', 'nikl_fv17', 'nikl_fv18', 'nikl_fv19', 'nikl_fv20', 'nikl_fx01', 'nikl_fx02', 'nikl_fx03', 'nikl_fx04', 'nikl_fx05', 'nikl_fx06', 'nikl_fx07', 'nikl_fx08', 'nikl_fx09', 'nikl_fx10', 'nikl_fx11', 'nikl_fx12', 'nikl_fx13', 'nikl_fx14', 'nikl_fx15', 'nikl_fx16', 'nikl_fx17', 'nikl_fx18', 'nikl_fx19', 'nikl_fx20', 'nikl_fy01', 'nikl_fy02', 'nikl_fy03', 'nikl_fy04', 'nikl_fy05', 'nikl_fy06', 'nikl_fy07', 'nikl_fy08', 'nikl_fy09', 'nikl_fy10', 'nikl_fy11', 'nikl_fy12', 'nikl_fy13', 'nikl_fy14', 'nikl_fy16', 'nikl_fy17', 'nikl_fy18', 'nikl_fz05', 'nikl_fz06', 'nikl_mv01', 'nikl_mv02', 'nikl_mv03', 'nikl_mv04', 'nikl_mv05', 'nikl_mv06', 'nikl_mv07', 'nikl_mv08', 'nikl_mv09', 'nikl_mv10', 'nikl_mv11', 'nikl_mv12', 'nikl_mv13', 'nikl_mv14', 'nikl_mv15', 'nikl_mv16', 'nikl_mv17', 'nikl_mv18', 'nikl_mv19', 'nikl_mv20', 'nikl_mw01', 'nikl_mw02', 'nikl_mw03', 'nikl_mw04', 'nikl_mw05', 'nikl_mw06', 'nikl_mw07', 'nikl_mw08', 'nikl_mw09', 'nikl_mw10', 'nikl_mw11', 'nikl_mw13', 'nikl_mw14', 'nikl_mw15', 'nikl_mw16', 'nikl_mw17', 'nikl_mw18', 'nikl_mw19', 'nikl_mw20', 'nikl_my01', 'nikl_my02', 'nikl_my03', 'nikl_my04', 'nikl_my05', 'nikl_my06', 'nikl_my07', 'nikl_my08', 'nikl_my09', 'nikl_my10', 'nikl_my11', 'nikl_mz01', 'nikl_mz02', 'nikl_mz03', 'nikl_mz04', 'nikl_mz05', 'nikl_mz06', 'nikl_mz07', 'nikl_mz08', 'nikl_mz09']
LJ_SPEAKERS = ['lj_0']
KSS_SPEAKERS = ['kss_0']

SPEAKERS = LIBRI_SPEAKERS + ZEROTH_SPEAKERS + NIKL_SPEAKERS + LJ_SPEAKERS + KSS_SPEAKERS
# SPEAKERS = [LIBRI_SPEAKERS[0], ZEROTH_SPEAKERS[0], NIKL_SPEAKERS[0]]

SPEAKER_NAME2IDX = {n: i for i, n in enumerate(SPEAKERS)}
SPEAKER_IDX2NAME = {i: n for i, n in enumerate(SPEAKERS)}

LIBRI_ROOT = "E:\\train-clean-100\\LibriSpeech\\train-clean-100"
ZEROTH_ROOT = "E:\\zeroth_korean\\train_data_01\\003"
NIKL_ROOT = "E:\\NIKL"
LJ_ROOT = "E:\\LJSpeech-1.1"
KSS_ROOT = "E:\\korean-single-speaker-speech-dataset"


def get_files(speaker):
    speaker_name = speaker.split("_")[1]
    current_speaker_data = []

    if speaker.startswith("libri_"):
        for chapter in os.listdir(os.path.join(LIBRI_ROOT, speaker_name)):
            txt = open(os.path.join(LIBRI_ROOT, speaker_name, chapter, speaker_name + "-" + chapter + ".trans.txt"), mode="r", encoding="utf-8")
            lines = txt.readlines()
            for line in lines:
                wav_file = line.split(" ")[0]
                wav_file = os.path.join(LIBRI_ROOT, speaker_name, chapter, wav_file + ".flac")

                if os.path.isfile(wav_file):
                    current_speaker_data.append([wav_file, vocabulary.str2token(" ".join(line.split(" ")[1:])), SPEAKER_NAME2IDX[speaker]])
                else:
                    print("파일이 존재하지 않습니다:", wav_file)

    if speaker.startswith("zeroth_"):
        txt = open(os.path.join(ZEROTH_ROOT, speaker_name, speaker_name + "_003" + ".trans.txt"), mode="r", encoding="utf-8")
        lines = txt.readlines()
        for line in lines:
            wav_file = line.split(" ")[0]
            wav_file = os.path.join(ZEROTH_ROOT, speaker_name, wav_file + ".flac")

            if os.path.isfile(wav_file):
                current_speaker_data.append([wav_file, vocabulary.str2token(" ".join(line.split(" ")[1:])), SPEAKER_NAME2IDX[speaker]])
            else:
                print("파일이 존재하지 않습니다:", wav_file)

    if speaker.startswith("nikl_"):
        txt = open(os.path.join(NIKL_ROOT, "script_nmbd_by_sentence_fixed.txt"), mode="r", encoding="utf-16")
        lines = txt.readlines()[21:]
        all_sentences = []
        current_chapter_sentences = []

        for line in lines:
            line = line.strip()

            if not line:
                continue
            if line.startswith("<" + str(len(all_sentences) + 2)):
                all_sentences.append(current_chapter_sentences)
                current_chapter_sentences = []
                continue

            dot_idx = line.find(".")
            current_chapter_sentences.append(line[dot_idx + 1:].strip())

        all_sentences.append(current_chapter_sentences)

        for wav_file in os.listdir(os.path.join(NIKL_ROOT, speaker_name)):
            if not wav_file.endswith(".wav"):
                continue
            try:
                chapter = int(wav_file.split(".")[0].split("_")[1][1:]) - 1
                sentence = int(wav_file.split(".")[0].split("_")[2][1:]) - 1

                wav_file = os.path.join(NIKL_ROOT, speaker_name, wav_file)

                current_speaker_data.append([wav_file, vocabulary.str2token(all_sentences[chapter][sentence]), SPEAKER_NAME2IDX[speaker]])
            except:
                print("다음 파일을 무시합니다:", wav_file)

    if speaker.startswith("lj_"):
        csv_file = open(os.path.join(LJ_ROOT, "metadata.csv"), mode="r", encoding="utf-8")
        lines = csv_file.readlines()
        for line in lines:
            spl = line.split("|")
            wav_file = os.path.join(LJ_ROOT, "wavs", spl[0] + ".wav")
            if os.path.isfile(wav_file):
                current_speaker_data.append([wav_file, vocabulary.str2token(spl[2]), SPEAKER_NAME2IDX[speaker]])
            else:
                print("파일이 존재하지 않습니다:", wav_file)

    if speaker.startswith("kss_"):
        csv_file = open(os.path.join(KSS_ROOT, "transcript.v.1.3.txt"), mode="r", encoding="utf-8")
        lines = csv_file.readlines()
        for line in lines:
            spl = line.split("|")
            wav_file = os.path.join(KSS_ROOT, "kss", spl[0])
            if os.path.isfile(wav_file):
                current_speaker_data.append([wav_file, vocabulary.str2token(spl[3]), SPEAKER_NAME2IDX[speaker]])
            else:
                print("파일이 존재하지 않습니다:", wav_file)

    data = []
    for speaker_data in current_speaker_data:
        try:
            wav, sr = librosa.load(speaker_data[0], sr=None)  # 오디오 길이 어림을 위해 파일 자체의 sr 사용 (리샘플링이 없으므로 빠름)
            wav = audio.fix_audio(wav, sr)
            data.append([speaker_data[0], speaker_data[1], speaker_data[2], wav.shape[0] / sr])
        except:
            print("다음 파일을 무시합니다:", speaker_data[0])

    return data


def get_all_files():
    with Pool(8) as pool:
        data = pool.map(get_files, SPEAKERS)

    data = [item for sublist in data for item in sublist]
    return data


def get_specific_files(speaker):
    return get_files(speaker)
