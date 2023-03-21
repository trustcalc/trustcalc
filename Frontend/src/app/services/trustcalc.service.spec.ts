import { TestBed } from '@angular/core/testing';

import { trustcalcService } from './trustcalc.service';

describe('trustcalcService', () => {
  let service: trustcalcService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(trustcalcService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
