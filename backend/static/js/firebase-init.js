(function(){
  // Firebase config provided by user
  const firebaseConfig = {
    apiKey: "AIzaSyA6yiIWclOlia4pTH-dUaSAW0xNh2rNhUY",
    authDomain: "vocalscan.firebaseapp.com",
    projectId: "vocalscan",
    storageBucket: "vocalscan.firebasestorage.app",
    messagingSenderId: "1021463923814",
    appId: "1:1021463923814:web:971f5a74d85fb3ce3ab1a7"
  };

  firebase.initializeApp(firebaseConfig);
  const auth = firebase.auth();
  const db = firebase.firestore();
  const storage = firebase.storage();

  // Ensure correct bucket root (Firebase config may omit legacy appspot bucket)
  const defaultBucketUrl = `gs://vocalscan.appspot.com`;
  const storageRootRef = storage.refFromURL(defaultBucketUrl);

  window.vsFirebase = { auth, db, storage, storageRootRef };
})();


