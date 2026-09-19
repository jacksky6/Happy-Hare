import unittest

from collections import deque

from test.hh.bootstrap import install

install()

from extras.mmu.unit import mmu_nfc_manager
from extras.mmu.unit.mmu_nfc_manager import MmuNfcManager
from extras.mmu.unit.nfc.mmu_nfc_reader import MmuNfcReader
from extras.mmu.unit.nfc.pn532_driver import (
    PN532_ACK,
    PN532Driver,
    PN532I2CStatusError,
)


class FakeTransferCommand:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def send(self, args, minclock=0, reqclock=0, retry=True):
        self.calls.append((list(args), retry))
        return self.responses.popleft()


class NewKlipperI2C:
    def __init__(self, responses):
        self.i2c_transfer_cmd = FakeTransferCommand(responses)

    def get_oid(self):
        return 17

    def i2c_read(self, write, read_len):
        raise AssertionError('new Klipper path used the shutdown-on-error wrapper')

    def i2c_write(self, write):
        raise AssertionError('new Klipper path used the shutdown-on-error wrapper')


class LegacyI2C:
    def __init__(self):
        self.writes = []
        self.reads = []

    def i2c_write(self, write):
        self.writes.append(list(write))

    def i2c_read(self, write, read_len):
        self.reads.append((list(write), read_len))
        return {'response': [0x01, 0xAB]}


def success(response=()):
    return {'i2c_bus_status': 'SUCCESS', 'response': list(response)}


def failure(status):
    return {'i2c_bus_status': status, 'response': []}


def status_error(status='START_READ_NACK', phase='response_status'):
    return PN532I2CStatusError(
        status=status,
        reader_name='gate3',
        operation='probe',
        command_code=0x4A,
        phase=phase,
        direction='read',
        read_len=1,
        probe_stage='response')


class TestSafeI2CTransfer(unittest.TestCase):
    def test_new_klipper_uses_raw_command_and_reports_all_statuses(self):
        for status in ('START_NACK', 'START_READ_NACK', 'NACK', 'BUS_TIMEOUT'):
            with self.subTest(status=status):
                i2c = NewKlipperI2C([failure(status)])
                driver = PN532Driver(i2c, 'gate3', debug=0)
                with self.assertRaises(PN532I2CStatusError) as raised:
                    driver._i2c_transfer_safe(
                        [], 1, phase='response_status', operation='probe',
                        command_code=0x4A)

                error = raised.exception
                self.assertEqual(error.status, status)
                self.assertEqual(error.reader_name, 'gate3')
                self.assertEqual(error.operation, 'probe')
                self.assertEqual(error.command, 'InListPassiveTarget')
                self.assertEqual(error.command_code, 0x4A)
                self.assertEqual(error.phase, 'response_status')
                self.assertEqual(error.direction, 'read')
                self.assertEqual(error.write_len, 0)
                self.assertEqual(error.read_len, 1)
                self.assertEqual(i2c.i2c_transfer_cmd.calls,
                                 [([17, [], 1], True)])

    def test_each_protocol_phase_keeps_command_context(self):
        stages = {
            'command_write': [],
            'ack_status': [success()],
            'ack_frame': [success(), success([0x01])],
            'response_status': [
                success(), success([0x01]), success([0x01] + PN532_ACK)],
            'response_frame': [
                success(), success([0x01]), success([0x01] + PN532_ACK),
                success([0x01])],
        }
        for expected_phase, prefix in stages.items():
            with self.subTest(phase=expected_phase):
                i2c = NewKlipperI2C(prefix + [failure('BUS_TIMEOUT')])
                driver = PN532Driver(i2c, 'gate3', debug=0)
                with self.assertRaises(PN532I2CStatusError) as raised:
                    driver._transceive(
                        [0x4A, 0x01, 0x00], 0x4B,
                        operation='read_target')

                error = raised.exception
                self.assertEqual(error.phase, expected_phase)
                self.assertEqual(error.command, 'InListPassiveTarget')
                self.assertEqual(error.command_code, 0x4A)
                self.assertEqual(error.operation, 'read_target')
                self.assertEqual(len(i2c.i2c_transfer_cmd.calls), len(prefix) + 1)

    def test_legacy_klipper_falls_back_to_unchanged_high_level_api(self):
        i2c = LegacyI2C()
        driver = PN532Driver(i2c, 'legacy', debug=0)

        self.assertFalse(driver.i2c_status_supported)
        self.assertEqual(driver._i2c_transfer_safe([0xAA], phase='command_write'), [])
        self.assertEqual(
            driver._i2c_transfer_safe([], 2, phase='response_frame'),
            [0x01, 0xAB])
        self.assertEqual(i2c.writes, [[0xAA]])
        self.assertEqual(i2c.reads, [([], 2)])

    def test_read_tag_does_not_hide_a_structured_i2c_error(self):
        driver = PN532Driver(LegacyI2C(), 'gate3', debug=0)
        expected = status_error()

        def fail_read_target(timeout=None):
            raise expected

        driver.read_target = fail_read_target
        with self.assertRaises(PN532I2CStatusError) as raised:
            driver.read_tag(timeout=0.1)
        self.assertIs(raised.exception, expected)


class FakeReactor:
    NOW = 0.0
    NEVER = 1e30

    def __init__(self, now=100.0):
        self.now = now

    def monotonic(self):
        return self.now

    def update_timer(self, timer, waketime):
        return


class ErrorChip:
    def __init__(self, error):
        self.error = error

    def read_tag(self, timeout=0.5):
        raise self.error


class TestReaderErrorState(unittest.TestCase):
    def test_reader_records_error_and_rethrows_it(self):
        error = status_error()
        reader = MmuNfcReader.__new__(MmuNfcReader)
        reader.reader = ErrorChip(error)
        reader.reactor = FakeReactor(123.5)
        reader.alive = True
        reader.present = False
        reader.last_uid = None
        reader.last_target_info = None
        reader.last_error = None
        reader.last_error_time = None
        reader.i2c_error_count = 0
        reader._last_error_object = None
        reader.reader_type = 'pn532'
        reader.interface = 'i2c'

        with self.assertRaises(PN532I2CStatusError) as raised:
            reader.read_uid(timeout=0.1)

        self.assertIs(raised.exception, error)
        self.assertFalse(reader.alive)
        self.assertEqual(reader.i2c_error_count, 1)
        self.assertEqual(reader.last_error_time, 123.5)
        self.assertEqual(reader.last_error['reader_name'], 'gate3')
        self.assertEqual(reader.last_error['status'], 'START_READ_NACK')
        self.assertEqual(reader.last_error['command'], 'InListPassiveTarget')
        self.assertEqual(reader.last_error['phase'], 'response_status')
        self.assertEqual(reader.get_status()['i2c_error_count'], 1)


class FakeMmu:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.debug = []

    def log_warning(self, message):
        self.warnings.append(message)

    def log_error(self, message):
        self.errors.append(message)

    def log_debug(self, message):
        self.debug.append(message)


class ManagerReader:
    def __init__(self, outcomes):
        self.name = 'gate3'
        self.gate = 3
        self.outcomes = deque(outcomes)
        self.probe_starts = 0
        self.recorded = []

    def record_communication_error(self, error):
        self.recorded.append(error)

    def clear_uid(self):
        return

    def read_uid(self, timeout=0.1):
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def probe_poll(self):
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def probe_start(self):
        self.probe_starts += 1
        return True

    def has_probe_support(self):
        return True


class FakeEndstop:
    def __init__(self, reader):
        self.reader = reader
        self.gate = 3
        self.triggers = []

    def trigger_handler(self, eventtime, triggered):
        self.triggers.append((eventtime, triggered))


def manager_with(reader):
    manager = MmuNfcManager.__new__(MmuNfcManager)
    manager.reactor = FakeReactor()
    manager.mmu = FakeMmu()
    manager._homing_poll_timer = object()
    manager._homing_endstop = None
    manager._homing_probe_error = None
    manager.is_enabled = lambda gate: True
    return manager


class TestManagerErrorBoundary(unittest.TestCase):
    def test_normal_read_becomes_warning_and_no_tag(self):
        error = status_error(status='NACK')
        reader = ManagerReader([error])
        manager = manager_with(reader)

        self.assertEqual(manager._read_reader(reader), (None, None))
        self.assertEqual(manager.mmu.errors, [])
        self.assertEqual(len(manager.mmu.warnings), 1)
        warning = manager.mmu.warnings[0]
        self.assertIn("reader 'gate3'", warning)
        self.assertIn('NACK', warning)
        self.assertIn('InListPassiveTarget(0x4A).response_status', warning)
        self.assertIn('I2C read_len=1', warning)

    def test_homing_error_triggers_endstop_without_retry(self):
        error = status_error(status='BUS_TIMEOUT')
        reader = ManagerReader([error, False])
        endstop = FakeEndstop(reader)
        manager = manager_with(reader)
        manager._homing_endstop = endstop

        next_time = manager._homing_poll(20.0)

        self.assertEqual(next_time, manager.reactor.NEVER)
        self.assertEqual(reader.probe_starts, 0,
                         'a bus error must not cause an immediate application retry')
        self.assertEqual(endstop.triggers, [(20.0, True)])
        self.assertIsNone(manager._homing_endstop)
        self.assertEqual(manager.mmu.errors, [])
        self.assertIn('Stopping NFC homing as if its endstop was triggered;',
                      manager.mmu.warnings[0])

    def test_homing_start_error_triggers_on_first_timer_tick_without_i2c_retry(self):
        error = status_error(status='START_NACK', phase='command_write')
        reader = ManagerReader([])
        reader.probe_start = lambda: (_ for _ in ()).throw(error)
        endstop = FakeEndstop(reader)
        manager = manager_with(reader)

        manager.start_homing_poll(endstop)

        self.assertIs(manager._homing_probe_error, error)
        self.assertEqual(reader.probe_starts, 0)
        self.assertEqual(manager._homing_poll(21.0), manager.reactor.NEVER)
        self.assertEqual(endstop.triggers, [(21.0, True)])
        self.assertEqual(len(manager.mmu.warnings), 1)


if __name__ == '__main__':
    unittest.main()
