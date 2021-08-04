{ lib
, buildPythonPackage
, fetchPypi
, google-api-core
}:
buildPythonPackage rec {
  pname = "google-cloud-core";
  version = "1.3.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1n19q57y4d89cjgmrg0f2a7yp7l1np2448mrhpndq354h389m3w7";
  };
  propagatedBuildInputs = [ google-api-core ];
  checkInputs = [  ];
  doCheck = false;
}
