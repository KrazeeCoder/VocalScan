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

  // Use the default bucket from config as-is (supports firebasestorage.app buckets)
  const app = firebase.app();
  const options = app.options || {};
  const bucket = options.storageBucket;
  const storageRootRef = storage.ref();
  console.log('[VocalScan] Using storage bucket:', bucket || '(default)');

  // Expose handles
  window.vsFirebase = { auth, db, storage, storageRootRef };
})();


