{ lib
, buildPythonPackage
, fetchPypi
, six
, google-resumable-media
, google-cloud-core
}:
buildPythonPackage rec {
  pname = "google-cloud-bigquery";
  version = "1.24.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1ca22hzql8x1z6bx9agidx0q09w24jwzkgg49k5j1spcignwxz3z";
  };
  propagatedBuildInputs = [ six google-resumable-media google-cloud-core ];
  checkInputs = [  ];
  doCheck = false;
}
