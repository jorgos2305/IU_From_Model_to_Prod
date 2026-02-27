-- When running this script from scratch all data is deleted

CREATE SCHEMA IF NOT EXISTS `turbine`;
USE `turbine` ;

-- Table `turbine`.`prediction`
DROP TABLE IF EXISTS `turbine`.`prediction` ;

CREATE TABLE IF NOT EXISTS `turbine`.`prediction` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `station_id` VARCHAR(50) NOT NULL,
  `ts` DATETIME NOT NULL,
  `temperature` DECIMAL(10,4) NOT NULL,
  `humidity` DECIMAL(10,4) NOT NULL,
  `noise` DECIMAL(10,4) NOT NULL,
  `is_anomaly` BOOL NOT NULL,
  `y_true` BOOL NOT NULL,
  `created_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`));
    