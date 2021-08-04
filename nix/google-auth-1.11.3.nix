{ lib
, buildPythonPackage
, fetchPypi
, six
, rsa
, pyasn1-modules
, cachetools
}:
buildPythonPackage rec {
  pname = "google-auth";
  version = "1.11.3";
  src = fetchPypi {
    inherit pname version;
    sha256 = "05av4clwv7kdk1v55ibcv8aim6dwfg1mi4wy0vv91fr6wq3205zc";
  };
  propagatedBuildInputs = [ six rsa pyasn1-modules cachetools ];
  checkInputs = [  ];
  doCheck = false;
}
