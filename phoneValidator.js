function telephoneCheck(str) {
let regex=/^(1\s?)?((\(\d{3}\)\s?)|(\d{3}[\s-]?))(\d{3}[\s-]?)(\d{4})$/
try{
    str.match(regex)[0];
    return true;
}
catch{return false;}}

telephoneCheck("1 555-555-5555");