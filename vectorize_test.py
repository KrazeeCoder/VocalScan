from backend.models.voice_features import UCIPDVectorizer, vectorize_wav
import os, sys, traceback

csv = os.path.join('ml_files','datasets','voice_recordings','parkinsons.data')
print('loading vectorizer from:', csv)
vec = UCIPDVectorizer.from_csv(csv)
print('vectorizer ready, dim =', len(vec.header))

test = r'C:\Windows\Media\Windows Notify System Generic.wav'
if len(sys.argv) > 1:
    test = sys.argv[1]
print('testing wav:', test)

try:
    x = vectorize_wav(test, vec)
    print('vectorize ok:', x.shape)
except Exception:
    traceback.print_exc()
    sys.exit(1)
