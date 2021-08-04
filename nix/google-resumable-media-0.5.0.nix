{ lib
, buildPythonPackage
, fetchPypi
, six
}:
buildPythonPackage rec {
  pname = "google-resumable-media";
  version = "0.5.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "0aldswz9lsw05a2gx26yjal6lcxhfqpn085zk1czvjz1my4d33ra";
  };
  propagatedBuildInputs = [ six ];
  checkInputs = [  ];
  doCheck = false;
}
