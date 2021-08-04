{ lib
, buildPythonPackage
, fetchPypi
}:
buildPythonPackage rec {
  pname = "mistletoe";
  version = "0.7.2";
  src = fetchPypi {
    inherit pname version;
    sha256 = "18z6hqfnfjqnrcgfgl5pkj9ggf9yx0yyy94azcn1qf7hqn6g3l14";
  };
  propagatedBuildInputs = [];
  checkInputs = [  ];
  doCheck = false;
}
