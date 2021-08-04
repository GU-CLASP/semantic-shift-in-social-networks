{ lib
, buildPythonPackage
, fetchPypi
, six
, google-resumable-media
, google-cloud-core
, google-api-core
}:
buildPythonPackage rec {
  pname = "google-cloud-bigquery-storage";
  version = "0.8.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1rbrx0xfy7v503x8mrc4v32758g2xjzn4b3d97vficmis4rf7frp";
  };
  propagatedBuildInputs = [ six google-api-core ];
  checkInputs = [  ];
  doCheck = false;
}
